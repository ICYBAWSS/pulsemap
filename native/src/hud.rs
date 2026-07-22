//! On-screen text + panel HUD: progress/ETA, hover banner, section legend.
//! Screen-space (not affected by the world camera). Rects use a small custom
//! instanced pipeline (rounded, anti-aliased); text uses glyphon/cosmic-text
//! for real font shaping, kerning, and subpixel AA.

use bytemuck::{Pod, Zeroable};
use glyphon::{
    Attrs, Buffer as TextBuffer, Cache, Color as GColor, Family, FontSystem, Metrics, Resolution,
    Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport, Weight,
};

use crate::layout::SectionMeta;

// ---- palette (matches the web UI's CSS custom properties) ------------------
pub const COL_PANEL: [f32; 4] = [0.0902, 0.1020, 0.1294, 0.82]; // --panel, translucent
pub const COL_TEXT: GColor = GColor::rgb(230, 232, 236); // --text
pub const COL_DIM: GColor = GColor::rgb(138, 146, 158); // --dim
pub const COL_BORDER: [f32; 4] = [0.1647, 0.1843, 0.2275, 1.0]; // border
pub const COL_ACCENT: [f32; 4] = [0.2392, 0.4196, 1.0, 1.0]; // #3d6bff

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RectInstance {
    pos: [f32; 2],
    size: [f32; 2],
    radius: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScreenUniform {
    size: [f32; 2],
    _pad: [f32; 2],
}

const RECT_QUAD: &[[f32; 2]] = &[
    [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
    [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
];
const MAX_RECTS: usize = 256;

/// One panel-like rect to draw, in physical-pixel screen space (top-left origin).
pub struct Rect {
    pub pos: [f32; 2],
    pub size: [f32; 2],
    pub radius: f32,
    pub color: [f32; 4],
}

/// One label to draw, in physical-pixel screen space. Owns its text (built
/// fresh each frame via `format!`) rather than borrowing, so callers don't
/// need to fight lifetimes over temporary strings.
pub struct Label {
    pub text: String,
    pub pos: [f32; 2],
    pub size_px: f32,
    pub color: GColor,
    pub bold: bool,
    /// Clip bounds (right/bottom edge); pass a huge value to effectively disable.
    pub max_width: f32,
}

pub struct Hud {
    font_system: FontSystem,
    swash_cache: SwashCache,
    _cache: Cache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,

    rect_pipeline: wgpu::RenderPipeline,
    screen_buf: wgpu::Buffer,
    rect_bind_group: wgpu::BindGroup,
    rect_quad_buf: wgpu::Buffer,
    rect_instance_buf: wgpu::Buffer,

    /// Shaped-text cache keyed by (text, size bits, bold, max_width bits).
    /// The HUD redraws continuously, and re-shaping every label every frame was
    /// stealing main-thread CPU from the analysis workers.
    text_cache: std::collections::HashMap<(String, u32, bool, u32), TextBuffer>,
}

/// Shape text into a laid-out buffer. A free function rather than a method so
/// the draw path can hold `&mut font_system` and `&text_cache` as disjoint
/// borrows of `Hud` simultaneously.
fn shape(fs: &mut FontSystem, text: &str, size_px: f32, bold: bool, max_width: f32) -> TextBuffer {
    let mut buf = TextBuffer::new(fs, Metrics::new(size_px, size_px * 1.3));
    buf.set_size(Some(max_width), Some(size_px * 4.0));
    let mut attrs = Attrs::new().family(Family::SansSerif);
    if bold {
        attrs = attrs.weight(Weight::BOLD);
    }
    buf.set_text(text, &attrs, Shaping::Advanced, None);
    buf.shape_until_scroll(fs, false);
    buf
}

impl Hud {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

        // ---- rect pipeline ----
        let screen_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-screen"),
            size: std::mem::size_of::<ScreenUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud-rect-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let rect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud-rect-bind"),
            layout: &rect_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: screen_buf.as_entire_binding() }],
        });
        let rect_quad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-rect-quad"),
            size: std::mem::size_of_val(RECT_QUAD) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&rect_quad_buf, 0, bytemuck::cast_slice(RECT_QUAD));
        let rect_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-rect-instances"),
            size: (std::mem::size_of::<RectInstance>() * MAX_RECTS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud-rect"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rect.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud-rect-pipeline-layout"),
            bind_group_layouts: &[Some(&rect_layout)],
            immediate_size: 0,
        });
        let rect_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud-rect-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<RectInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            1 => Float32x2, 2 => Float32x2, 3 => Float32, 4 => Float32x4
                        ],
                    }),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            font_system,
            swash_cache,
            _cache: cache,
            viewport,
            atlas,
            text_renderer,
            rect_pipeline,
            screen_buf,
            rect_bind_group,
            rect_quad_buf,
            rect_instance_buf,
            text_cache: std::collections::HashMap::new(),
        }
    }

    pub fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.viewport.update(queue, Resolution { width, height });
        let u = ScreenUniform { size: [width as f32, height as f32], _pad: [0.0; 2] };
        queue.write_buffer(&self.screen_buf, 0, bytemuck::bytes_of(&u));
    }

    fn make_buffer(&mut self, text: &str, size_px: f32, bold: bool, max_width: f32) -> TextBuffer {
        shape(&mut self.font_system, text, size_px, bold, max_width)
    }

    /// Drop cached shaped text (e.g. after a DPI change makes every size stale).
    pub fn clear_text_cache(&mut self) {
        self.text_cache.clear();
    }

    /// Rendered width of `text` in px — so a button can be sized to its label
    /// instead of a guessed `len() * size * 0.6` fudge.
    pub fn measure(&mut self, text: &str, size_px: f32, bold: bool) -> f32 {
        let buf = self.make_buffer(text, size_px, bold, 10_000.0);
        buf.layout_runs().map(|r| r.line_w).fold(0.0, f32::max)
    }

    /// Draw a frame's worth of rects + labels. `viewport_size` is physical px.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        rects: &[Rect],
        labels: &[Label],
    ) {
        // ---- rects ----
        let n = rects.len().min(MAX_RECTS);
        if n > 0 {
            let insts: Vec<RectInstance> = rects[..n]
                .iter()
                .map(|r| RectInstance { pos: r.pos, size: r.size, radius: r.radius, color: r.color })
                .collect();
            queue.write_buffer(&self.rect_instance_buf, 0, bytemuck::cast_slice(&insts));
            pass.set_pipeline(&self.rect_pipeline);
            pass.set_bind_group(0, &self.rect_bind_group, &[]);
            pass.set_vertex_buffer(0, self.rect_quad_buf.slice(..));
            pass.set_vertex_buffer(1, self.rect_instance_buf.slice(..));
            pass.draw(0..RECT_QUAD.len() as u32, 0..n as u32);
        }

        // ---- text ----
        if labels.is_empty() {
            return;
        }
        // Crude bound: transient strings (the ETA ticks every frame) would grow
        // this without limit. Evict BEFORE filling — doing it between the fill
        // and the lookup meant the evicting frame missed every key and panicked.
        if self.text_cache.len() > 400 {
            self.text_cache.clear();
        }
        // Shaping is expensive and the HUD redraws continuously (streaming
        // analysis, force sim, waveform). Almost every label — legend rows,
        // title, buttons — is byte-identical frame to frame, so shape once and
        // reuse. Only genuinely changing text (progress, ETA, hover) re-shapes.
        for l in labels {
            let key = (l.text.clone(), l.size_px.to_bits(), l.bold, l.max_width.to_bits());
            if !self.text_cache.contains_key(&key) {
                let buf = shape(&mut self.font_system, &l.text, l.size_px, l.bold, l.max_width);
                self.text_cache.insert(key, buf);
            }
        }
        // `get` rather than indexing: every key was just inserted above, but a
        // cache miss should drop one label for one frame, never kill the app.
        let areas: Vec<TextArea> = labels
            .iter()
            .filter_map(|l| {
                let key = (l.text.clone(), l.size_px.to_bits(), l.bold, l.max_width.to_bits());
                self.text_cache.get(&key).map(|buf| (buf, l))
            })
            .map(|(buf, l)| TextArea {
                buffer: buf,
                left: l.pos[0],
                top: l.pos[1],
                scale: 1.0,
                bounds: TextBounds {
                    left: l.pos[0] as i32,
                    top: l.pos[1] as i32,
                    right: (l.pos[0] + l.max_width) as i32,
                    bottom: (l.pos[1] + l.size_px * 4.0) as i32,
                },
                default_color: l.color,
                custom_glyphs: &[],
            })
            .collect();

        if self
            .text_renderer
            .prepare(device, queue, &mut self.font_system, &mut self.atlas, &self.viewport, areas, &mut self.swash_cache)
            .is_ok()
        {
            let _ = self.text_renderer.render(&self.atlas, &self.viewport, pass);
        }
        self.atlas.trim();
    }
}

/// Screen-space layout for the legend panel (top-right): one row per section.
/// Shared between drawing and hit-testing so they can never drift apart.
pub struct LegendRow {
    pub rect: [f32; 4], // x, y, w, h (px)
    pub swatch: [f32; 4],
    pub section: String,
    pub color: [f32; 4],
    pub count: usize,
}

/// `scale` is the window's DPI scale factor — all coordinates here are
/// physical px, so this must match what's used to size the text/rects drawn
/// on top, or the legend's clickable area drifts from what's rendered.
pub fn legend_layout(sections: &[SectionMeta], viewport_w: f32, scale: f32) -> Vec<LegendRow> {
    let row_h = 26.0 * scale;
    let panel_w = 220.0 * scale;
    let pad = 10.0 * scale;
    let x = viewport_w - panel_w - 16.0 * scale;
    let mut y = (16.0 + 34.0) * scale; // below the "Sections" header
    let mut rows = Vec::new();
    for s in sections {
        let rect = [x, y, panel_w, row_h];
        let swatch = [x + pad, y + row_h / 2.0 - 5.0 * scale, 10.0 * scale, 10.0 * scale];
        rows.push(LegendRow {
            rect,
            swatch,
            section: s.name.clone(),
            color: [s.color[0], s.color[1], s.color[2], 1.0],
            count: s.count,
        });
        y += row_h;
    }
    rows
}
