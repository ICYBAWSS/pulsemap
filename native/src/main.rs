// PulseMap native GUI. Drop a folder → background thread decodes/embeds/
// classifies each sound (reusing the verified pipeline, on-disk cached) →
// lays them out by section → streams the result into the GPU point cloud.
// Pan/zoom, hover-audition, search, legend, and drag-to-reclassify.
//
// Still not full parity: no continuous force-sim "breathing" physics yet
// (layout is static collision-relaxed at build time) — see NATIVE_APP_PLAN.md.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use pulsemap::analyzer::{collect_audio, Analyzer, WAVE_BARS};
use pulsemap::hud::{self, Hud};
use pulsemap::layout::{build_map, Node, SectionMeta};
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

/// Runtime asset folder holding `audio_model.onnx`, `mel_slaney.npy` and
/// `model.json`. Next to the executable in a distributed build, or in
/// `Contents/Resources` inside a macOS .app (only code may live in
/// `Contents/MacOS`, or codesign refuses to seal the bundle), falling back to
/// the source tree when running from cargo. A compiled-in absolute path only
/// resolves on the machine that built the binary, so it can't be shipped.
fn asset_dir() -> PathBuf {
    let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from));
    let candidates = exe_dir.into_iter().flat_map(|d| {
        [d.join("models"), d.join("../Resources/models")]
    });
    for p in candidates {
        if p.join("audio_model.onnx").is_file() {
            return p;
        }
    }
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/models"))
}

// ---- GPU data ---------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    center: [f32; 2],
    viewport: [f32; 2],
    zoom: f32,
    point_px: f32,
    hovered: f32,         // instance index of hovered node, -1 = none
    hovered_section: f32, // section_id of hovered node's section, -1 = none
    time: f32,            // seconds since launch, drives the ambient drift
    _pad: [f32; 3],       // std140: pad to a 16-byte multiple
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointInstance {
    pos: [f32; 2],
    color: [f32; 3],
    confidence: f32,
    rel: [f32; 2],
    section_id: u32,
    dim: f32, // search-miss dim multiplier, applied to FINAL alpha in the shader
}

use pulsemap::layout::NODE_R;
const GREY: [f32; 3] = [0.357, 0.392, 0.447]; // #5b6472, matches app.js GREY
const INTRO_DURATION: Duration = Duration::from_millis(900);

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

const QUAD: &[[f32; 2]] = &[
    [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
    [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
];

// ---- analysis thread messages ----------------------------------------------

enum Msg {
    /// File count, sent before analysis begins so the grey blob can appear.
    Start { total: usize },
    Progress { done: usize, total: usize },
    /// One sound finished analysis — streamed so the map fills in sound by
    /// sound instead of appearing all at once when the folder completes.
    Sorted { filename: String, section: String, confidence: f32 },
    Done(Vec<Node>, Vec<SectionMeta>),
    Error(String),
}

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use pulsemap::cache::{fingerprint, Cache, CacheEntry};
use pulsemap::layout::Analyzed;

/// Per-user state folder: `~/.pulsemap`, or `%USERPROFILE%\.pulsemap` on
/// Windows, which has no HOME.
fn state_dir() -> PathBuf {
    match std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        Ok(home) => PathBuf::from(home).join(".pulsemap"),
        Err(_) => PathBuf::from("."),
    }
}

/// Where the on-disk embedding cache lives — the native replacement for the
/// web app's IndexedDB cache, so re-dropping an unchanged folder is instant.
fn cache_path() -> PathBuf {
    state_dir().join("cache.json")
}

/// Where drag-to-reclassify corrections are persisted — the native
/// replacement for the web app's manual "Export corrections" button: every
/// reclassification is saved immediately rather than requiring an export click.
fn corrections_path() -> PathBuf {
    state_dir().join("corrections.json")
}

/// A chunk of files for one worker, plus shared collectors for the whole job.
struct PoolJob {
    files: Vec<PathBuf>,
    results: Arc<Mutex<Vec<Analyzed>>>,
    cache: Arc<Mutex<Cache>>,
    done: Arc<AtomicUsize>,
    total: usize,
    progress: Sender<Msg>,
    complete: Sender<()>,
}

/// Block until every worker's session is built; returns the indices of those
/// that built successfully. Failures still report in, so this can't hang.
fn wait_ready(ready: &(Mutex<(usize, Vec<bool>)>, Condvar)) -> Vec<usize> {
    let (m, cv) = ready;
    let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
    while g.0 < g.1.len() {
        g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
    }
    g.1.iter().enumerate().filter(|(_, &ok)| ok).map(|(i, _)| i).collect()
}

/// Persistent pool of worker threads that each load the model ONCE at startup
/// and stay resident, so a folder-drop never pays the model-compile cost.
struct AnalyzerPool {
    workers: Vec<Sender<PoolJob>>,
    cache: Arc<Mutex<Cache>>,
    /// (settled count, per-worker success) + condvar. Workers report once their
    /// session is built; no job dispatches until every worker has settled.
    ///
    /// ORT's global type registry (`DataTypeImpl::GetDataType`) is read during
    /// session PLANNING *and* during inference, so a session being built while
    /// another thread runs inference is a data race — observed as a SIGSEGV in
    /// `PlannerImpl::GetElementSize` with sibling threads inside `Session::run`.
    /// The construction mutex in `Analyzer::new` only serialises build-vs-build;
    /// this barrier closes build-vs-inference.
    ready: Arc<(Mutex<(usize, Vec<bool>)>, Condvar)>,
}

impl AnalyzerPool {
    fn new(n: usize, assets: &Path, protos: &Path) -> Self {
        let ready = Arc::new((Mutex::new((0usize, vec![false; n])), Condvar::new()));
        let workers = (0..n)
            .map(|i| {
                let (tx, rx) = channel::<PoolJob>();
                let (assets, protos) = (assets.to_path_buf(), protos.to_path_buf());
                let ready = Arc::clone(&ready);
                thread::spawn(move || {
                    // Compiles/loads the model now, while the user picks a folder.
                    let built = Analyzer::new(&assets, &protos);
                    {
                        // Report settled even on failure, or the barrier hangs.
                        let (m, cv) = &*ready;
                        let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
                        g.0 += 1;
                        g.1[i] = built.is_ok();
                        cv.notify_all();
                    }
                    let mut analyzer = match built {
                        Ok(a) => a,
                        Err(_) => return,
                    };
                    while let Ok(job) = rx.recv() {
                        for f in &job.files {
                            let fp = fingerprint(f).ok();
                            let cached = fp
                                .as_ref()
                                .and_then(|k| job.cache.lock().unwrap().get(k).cloned());
                            let analyzed = if let Some(c) = cached {
                                // Cache hit: skip decode+embed+classify entirely.
                                Some(Analyzed {
                                    path: c.path,
                                    filename: c.filename,
                                    section: c.section,
                                    confidence: c.confidence,
                                    embedding: c.embedding,
                                    envelope: c.envelope,
                                    duration: c.duration,
                                })
                            } else {
                                let a = analyzer.analyze(f);
                                if let (Some(a), Some(k)) = (&a, &fp) {
                                    job.cache.lock().unwrap().insert(
                                        k.clone(),
                                        CacheEntry {
                                            path: a.path.clone(),
                                            filename: a.filename.clone(),
                                            embedding: a.embedding.clone(),
                                            section: a.section.clone(),
                                            confidence: a.confidence,
                                            tags: Vec::new(),
                                            envelope: a.envelope.clone(),
                                            duration: a.duration,
                                        },
                                    );
                                }
                                a
                            };
                            if let Some(a) = analyzed {
                                let _ = job.progress.send(Msg::Sorted {
                                    filename: a.filename.clone(),
                                    section: a.section.clone(),
                                    confidence: a.confidence,
                                });
                                job.results.lock().unwrap().push(a);
                            }
                            let n = job.done.fetch_add(1, Ordering::Relaxed) + 1;
                            let _ = job.progress.send(Msg::Progress { done: n, total: job.total });
                        }
                        let _ = job.complete.send(());
                    }
                });
                tx
            })
            .collect();
        AnalyzerPool { workers, cache: Arc::new(Mutex::new(Cache::load(cache_path()))), ready }
    }


    /// Fan a folder's files across the resident workers; emits progress + Done on `tx`.
    /// Analyze every folder in `folders` as one map. Adding a folder re-runs the
    /// whole set rather than merging layouts: the section layout is a global
    /// neighbour embedding, so a new sound can only be placed correctly by
    /// solving it alongside everything else. Re-analysis is cheap because every
    /// already-mapped file hits the embedding cache and skips decode+embed.
    ///
    /// `corrections` (path -> section) is applied *before* layout so a
    /// reclassified sound is laid out inside the section the user put it in,
    /// not the one the classifier picked.
    fn analyze_folders(
        &self,
        folders: Vec<PathBuf>,
        corrections: std::collections::HashMap<String, String>,
        tx: Sender<Msg>,
    ) {
        let all_workers = self.workers.clone();
        let cache = Arc::clone(&self.cache);
        let ready = Arc::clone(&self.ready);
        thread::spawn(move || {
            // Wait here, NOT on the caller: this runs on the UI thread's event
            // loop, and sessions can take seconds to build. Dropping a folder
            // during startup now waits instead of racing (or freezing the UI).
            let live = wait_ready(&ready);
            let workers: Vec<_> = live.iter().map(|&i| all_workers[i].clone()).collect();
            if workers.is_empty() {
                let _ = tx.send(Msg::Error("no analyzer could start".into()));
                return;
            }
            // Dedup: overlapping or re-added folders must not double up nodes.
            let mut files: Vec<PathBuf> = folders.iter().flat_map(|f| collect_audio(f)).collect();
            files.sort();
            files.dedup();
            if files.is_empty() {
                let _ = tx.send(Msg::Error("no audio files found".into()));
                return;
            }
            let total = files.len();
            // Announce the count up front so the UI can spawn the grey blob
            // before any file has been analyzed.
            let _ = tx.send(Msg::Start { total });
            let results = Arc::new(Mutex::new(Vec::new()));
            let done = Arc::new(AtomicUsize::new(0));
            let (ctx, crx) = channel::<()>();

            let chunk = total.div_ceil(workers.len()).max(1);
            let mut sent = 0;
            for (w, files_chunk) in files.chunks(chunk).enumerate() {
                let job = PoolJob {
                    files: files_chunk.to_vec(),
                    results: Arc::clone(&results),
                    cache: Arc::clone(&cache),
                    done: Arc::clone(&done),
                    total,
                    progress: tx.clone(),
                    complete: ctx.clone(),
                };
                let _ = workers[w].send(job);
                sent += 1;
            }
            for _ in 0..sent {
                let _ = crx.recv(); // wait for all chunks to finish
            }
            // Persist any new cache entries once per batch (not per-file: I/O cost).
            if let Err(e) = cache.lock().unwrap().save() {
                log::warn!("failed to save embedding cache: {e}");
            }
            let mut analyzed = std::mem::take(&mut *results.lock().unwrap());
            for a in &mut analyzed {
                if let Some(sec) = corrections.get(&a.path) {
                    a.section = sec.clone();
                }
            }
            let (nodes, sections) = build_map(analyzed);
            let _ = tx.send(Msg::Done(nodes, sections));
        });
    }
}

// ---- camera -----------------------------------------------------------------

struct Camera {
    center: [f32; 2],
    zoom: f32,
}

impl Camera {
    fn world_at(&self, cursor: (f64, f64), vp: (f32, f32)) -> [f32; 2] {
        let cx = cursor.0 as f32 - vp.0 * 0.5;
        let cy = vp.1 * 0.5 - cursor.1 as f32;
        [self.center[0] + cx / self.zoom, self.center[1] + cy / self.zoom]
    }
}

// ---- renderer ---------------------------------------------------------------

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    quad_buf: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    num_instances: u32,
    camera_buf: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera: Camera,
    cursor: (f64, f64),
    panning: bool,
    // interaction + audio
    nodes: Vec<Node>,
    hovered: Option<usize>,       // node under cursor (audition + grow)
    active_section: Option<u32>,  // category the cursor is within (gradient + grey others)
    audio: Option<rodio::MixerDeviceSink>,
    player: Option<rodio::Player>,
    // intro fly-in + colorize animation
    intro_start: Option<Instant>,
    intro_from: Vec<[f32; 2]>,
    // HUD: text/panels + section legend + live status
    hud: Hud,
    sections: Vec<SectionMeta>,
    hovered_legend: Option<usize>,
    status: Status,
    /// Window scale factor (e.g. 2.0 on Retina). Our surface/HUD coordinates
    /// are physical px, so every HUD size/position must be multiplied by this
    /// or it renders at half (or less) the intended visual size on HiDPI.
    ui_scale: f32,
    search_query: String,
    /// Measured px width of the Clear-cache label at the current ui_scale, so
    /// the button hugs its text. Refreshed on DPI change.
    clear_btn_w: f32,
    add_btn_w: f32,
    title_w: f32,
    open_btn_w: f32,
    /// (node index, when playback started) — drives the waveform playhead.
    playing: Option<(usize, Instant)>,
    /// Wall clock for time-based HUD animation.
    launched: Instant,
    /// Force-sim temperature. Cools to SIM_ALPHA_MIN and stops; a drag reheats
    /// it to 1.0 so the map re-settles.
    sim_alpha: f32,
    /// Allocated instance-buffer slots (grown geometrically while streaming).
    instance_cap: usize,
    /// How many blob nodes have been classified+coloured so far.
    colored: usize,
    // drag-to-reclassify
    dragging_node: Option<usize>,
    /// Where a node sat before its FIRST reclassification: section, spring
    /// anchor and gradient coords, keyed by file path. Dragging a sound back to
    /// the section it came from restores this exactly — otherwise it gets
    /// re-placed next to a neighbour and lands somewhere new every round trip.
    origins: std::collections::HashMap<String, (String, f32, f32, f32, f32)>,
    drag_target_section: Option<usize>,
    cmd_down: bool,
    /// Node pressed without Cmd, plus where the press landed: a plain drag
    /// hands the file to the OS (drop it straight into a DAW), so the gesture
    /// is armed here and only fires once the cursor actually moves — otherwise
    /// every click on a node would start a drag session.
    file_drag_arm: Option<(usize, (f64, f64))>,
    corrections: std::collections::HashMap<String, String>, // file path -> reassigned section
    toast: Option<(String, Instant)>,
}

/// Solid 32x32 BGR swatch as an uncompressed BMP — the preview that follows the
/// cursor on a drag-out. The drag crate hands its bytes to NSImage and only
/// accepts a file path or an encoded buffer, so *some* real image is required;
/// BMP is the one format simple enough to emit inline (no compression, no
/// checksum) and 32*3 bytes per row is already 4-byte aligned, so there's no
/// row padding to get wrong.
fn swatch_bmp(color: [f32; 3]) -> Vec<u8> {
    const N: u32 = 32;
    let px = N * N * 3;
    let mut b = Vec::with_capacity(54 + px as usize);
    b.extend(b"BM"); // ---- BITMAPFILEHEADER, 14 bytes
    b.extend((54 + px).to_le_bytes());
    b.extend([0u8; 4]); // reserved
    b.extend(54u32.to_le_bytes()); // pixel data offset
    b.extend(40u32.to_le_bytes()); // ---- BITMAPINFOHEADER, 40 bytes
    b.extend(N.to_le_bytes());
    b.extend(N.to_le_bytes());
    b.extend(1u16.to_le_bytes()); // planes
    b.extend(24u16.to_le_bytes()); // bits per pixel
    b.extend([0u8; 24]); // compression, sizes, resolutions, palette: all zero/default
    let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0) as u8;
    let bgr = [q(color[2]), q(color[1]), q(color[0])];
    for _ in 0..N * N {
        b.extend(bgr);
    }
    b
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// Mirrors the file-gathering in `analyze_folders`. "Add folder" re-runs the
    /// whole set, so a folder added twice — or one nested inside another — must
    /// not double up its files into duplicate nodes.
    fn gather(folders: &[PathBuf]) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> =
            folders.iter().flat_map(|f| pulsemap::analyzer::collect_audio(f)).collect();
        files.sort();
        files.dedup();
        files
    }

    #[test]
    fn adding_folders_unions_without_duplicates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../test_samples");
        if !root.exists() {
            return; // sample folder isn't checked in everywhere
        }
        let subs: Vec<PathBuf> = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        assert!(subs.len() >= 2, "need two sample folders to test a union");

        let (a, b) = (gather(&subs[..1]), gather(&subs[1..2]));
        assert!(!a.is_empty() && !b.is_empty());

        // Adding b to a: every file from both, counted once.
        let both = gather(&subs[..2]);
        assert_eq!(both.len(), a.len() + b.len());

        // Re-adding a folder already on the map changes nothing...
        assert_eq!(gather(&[subs[0].clone(), subs[0].clone()]).len(), a.len());
        // ...and neither does adding a parent that already contains it.
        assert_eq!(gather(&[root.clone(), subs[0].clone()]), gather(&[root]));
    }
    /// A malformed header here shows up as a panic inside NSImage at drag time,
    /// nowhere near this code — so pin the layout down.
    #[test]
    fn bmp_header_and_pixels() {
        let b = super::swatch_bmp([1.0, 0.5, 0.0]);
        assert_eq!(&b[0..2], b"BM");
        assert_eq!(b.len(), 54 + 32 * 32 * 3);
        assert_eq!(u32::from_le_bytes(b[2..6].try_into().unwrap()), b.len() as u32);
        assert_eq!(u32::from_le_bytes(b[10..14].try_into().unwrap()), 54); // pixel offset
        assert_eq!(u32::from_le_bytes(b[14..18].try_into().unwrap()), 40); // info header size
        assert_eq!(u16::from_le_bytes(b[28..30].try_into().unwrap()), 24); // bpp
        assert_eq!(&b[54..57], &[0, 127, 255]); // BGR, not RGB
    }
}

fn rect_hit([x, y, w, h]: [f32; 4], c: (f64, f64)) -> bool {
    c.0 as f32 >= x && c.0 as f32 <= x + w && c.1 as f32 >= y && c.1 as f32 <= y + h
}

const OPEN_BTN_LABEL: &str = "Choose folder…";
const SIM_ALPHA_MIN: f32 = 0.001; // force sim stops cooling below this
const WAVE_ATTACK: f32 = 0.09; // s: flat -> the sound's shape
const WAVE_RELEASE: f32 = 1.8; // s: shape -> flat once playback ends (slow settle)
const CLEAR_BTN_LABEL: &str = "Clear cache";
const ADD_BTN_LABEL: &str = "Add folder…";
const TITLE: &str = "PulseMap";
const TITLE_PX: f32 = 20.0; // unscaled top-left wordmark size
const DOCK_H: f32 = 44.0;      // unscaled height of both bottom docks
const DOCK_PAD: f32 = 16.0;    // unscaled margin from the window edge
const BTN_GAP: f32 = 6.0;      // unscaled inset of a button inside its dock
const INFO_DOCK_W: f32 = 440.0;
const WAVE_W: f32 = 150.0;     // unscaled waveform width inside the info dock
const BTN_TEXT_PX: f32 = 13.0; // unscaled; multiplied by ui_scale at use
const BTN_PAD_X: f32 = 14.0;   // unscaled horizontal padding either side of a button label
const DROP_RADIUS: f32 = 20.0; // world units; how close a drop must land to a section center
const TOAST_DURATION: Duration = Duration::from_millis(2500);

/// What the HUD's progress panel should show; owned by State so render() can
/// read it without threading extra params through the event loop.
enum Status {
    Idle,
    Analyzing { done: usize, total: usize, eta_secs: Option<u32> },
    Done { count: usize, took_secs: f32 },
}

impl State {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let ui_scale = window.scale_factor() as f32;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("no GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("no device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::default(),
        };
        surface.configure(&device, &config);

        let camera = Camera { center: [0.0, 0.0], zoom: 1.2 };
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-layout"),
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
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bind"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
        });

        let quad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quad"),
            size: std::mem::size_of_val(QUAD) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&quad_buf, 0, bytemuck::cast_slice(QUAD));

        // Empty instance buffer to start (drop a folder to fill it).
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (std::mem::size_of::<PointInstance>() * 1) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point"),
            source: wgpu::ShaderSource::Wgsl(include_str!("point.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point-pipeline"),
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
                        array_stride: std::mem::size_of::<PointInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            1 => Float32x2, 2 => Float32x3, 3 => Float32, 4 => Float32x2, 5 => Uint32, 6 => Float32
                        ],
                    }),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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

        let mut hud = Hud::new(&device, &queue, config.format);
        hud.resize(&queue, config.width, config.height);
        let clear_btn_w = hud.measure(CLEAR_BTN_LABEL, BTN_TEXT_PX * ui_scale, false);
        let add_btn_w = hud.measure(ADD_BTN_LABEL, BTN_TEXT_PX * ui_scale, false);
        let title_w = hud.measure(TITLE, TITLE_PX * ui_scale, true);
        // Startup screen renders this button larger; measure at that size so the
        // pill hugs the label in whichever state it's drawn.
        let open_btn_w = hud.measure(OPEN_BTN_LABEL, 15.0 * ui_scale, false);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            quad_buf,
            instance_buf,
            num_instances: 0,
            camera_buf,
            camera_bind_group,
            camera,
            cursor: (0.0, 0.0),
            panning: false,
            nodes: Vec::new(),
            hovered: None,
            active_section: None,
            audio: rodio::DeviceSinkBuilder::open_default_sink().ok(),
            player: None,
            intro_start: None,
            intro_from: Vec::new(),
            hud,
            sections: Vec::new(),
            hovered_legend: None,
            status: Status::Idle,
            ui_scale,
            search_query: String::new(),
            clear_btn_w,
            add_btn_w,
            title_w,
            open_btn_w,
            playing: None,
            launched: Instant::now(),
            sim_alpha: 0.0,
            instance_cap: 0,
            colored: 0,
            dragging_node: None,
            origins: std::collections::HashMap::new(),
            drag_target_section: None,
            cmd_down: false,
            file_drag_arm: None,
            corrections: {
                std::fs::read_to_string(corrections_path())
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default()
            },
            toast: None,
        }
    }

    /// Nearest node under the cursor within an ~8px hit radius (node world pos
    /// is [x, -y] to match the render flip).
    /// Nearest node to the cursor and its world-space distance.
    fn nearest_node(&self, cursor: (f64, f64)) -> Option<(usize, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let w = self.camera.world_at(cursor, self.viewport());
        let mut best = 0usize;
        let mut best_d2 = f32::MAX;
        for (i, n) in self.nodes.iter().enumerate() {
            let dx = w[0] - n.x;
            let dy = w[1] - (-n.y);
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        Some((best, best_d2.sqrt()))
    }

    /// Node exactly under the cursor (within its visible hit radius) — used to
    /// decide whether a press starts a node-drag vs a canvas pan.
    fn node_hit(&self, cursor: (f64, f64)) -> Option<usize> {
        let hit_r = NODE_R.max(7.0 / self.camera.zoom);
        self.nearest_node(cursor).filter(|&(_, d)| d < hit_r).map(|(i, _)| i)
    }

    /// Index of the legend row under the cursor (screen-space, physical px),
    /// if any. Shares layout with `Hud`'s draw call so hit-test and rendered
    /// position can never drift apart.
    /// The bottom-left button dock once a map is loaded: outer panel, plus one
    /// rect per button in [add, open, clear] order. One panel holding three
    /// labels instead of three free-floating pills — they're one group of
    /// actions and read as clutter stacked separately.
    ///
    /// Single source of truth for drawing AND hit-testing, so the two can't
    /// drift apart.
    fn btn_dock(&self) -> ([f32; 4], [[f32; 4]; 2]) {
        let s = self.ui_scale;
        let (_, vh) = self.viewport();
        let h = DOCK_H * s;
        let y = vh - h - DOCK_PAD * s;
        let mut x = (DOCK_PAD + BTN_GAP) * s;
        let mut items = [[0.0f32; 4]; 2];
        for (i, lw) in [self.add_btn_w, self.clear_btn_w].iter().enumerate() {
            let iw = lw + 2.0 * BTN_PAD_X * s;
            items[i] = [x, y + BTN_GAP * s, iw, h - 2.0 * BTN_GAP * s];
            x += iw + BTN_GAP * s;
        }
        ([DOCK_PAD * s, y, x - DOCK_PAD * s, h], items)
    }

    /// Bottom-right dock: the hovered/playing sound's name on the left, its
    /// waveform on the right, in ONE panel. They describe the same sound, so
    /// two separate boxes was just noise along the bottom edge.
    fn info_dock(&self) -> [f32; 4] {
        let s = self.ui_scale;
        let (vw, vh) = self.viewport();
        let h = DOCK_H * s;
        // Give up width rather than slide under the button dock when the window
        // is narrow; the label column truncates on its own.
        let left = self.btn_dock().0[2] + (DOCK_PAD + 8.0) * s;
        let w = (INFO_DOCK_W * s).min(vw - left - DOCK_PAD * s).max(WAVE_W * s);
        [vw - w - DOCK_PAD * s, vh - h - DOCK_PAD * s, w, h]
    }

    /// "Clear cache" button rect [x, y, w, h] in physical px. Sits in the bottom
    /// dock once a map is loaded; on the empty startup screen there is no dock,
    /// so it keeps its own bottom-left corner.
    fn clear_btn_rect(&self) -> [f32; 4] {
        if !self.on_startup_screen() {
            return self.btn_dock().1[1];
        }
        let s = self.ui_scale;
        let (_, vh) = self.viewport();
        let h = DOCK_H * s - 2.0 * BTN_GAP * s;
        let w = self.clear_btn_w + 2.0 * BTN_PAD_X * s;
        [16.0 * s, vh - h - 16.0 * s, w, h]
    }

    fn clear_btn_hit(&self, cursor: (f64, f64)) -> bool {
        rect_hit(self.clear_btn_rect(), cursor)
    }

    /// True while the empty dropzone screen is showing.
    fn on_startup_screen(&self) -> bool {
        self.nodes.is_empty() && matches!(self.status, Status::Idle)
    }

    /// "Choose folder…" button. Prominent and centred on the startup screen
    /// (where picking a folder is the only thing to do), tucked in above the
    /// Clear-cache button once a map is loaded.
    /// "Choose folder…" — startup screen only. Once a map is loaded "Add folder"
    /// covers picking more, and the title click starts over, so a second
    /// load-replacing button had nothing left to do.
    fn open_btn_rect(&self) -> [f32; 4] {
        let s = self.ui_scale;
        let w = self.open_btn_w + 2.0 * BTN_PAD_X * s;
        let (vw, vh) = self.viewport();
        [(vw - w) * 0.5, vh * 0.5 + 58.0 * s, w, 38.0 * s]
    }

    fn open_btn_hit(&self, cursor: (f64, f64)) -> bool {
        self.on_startup_screen() && rect_hit(self.open_btn_rect(), cursor)
    }

    /// "Add folder…" button, first slot in the dock. Only exists once a map is
    /// loaded — on the startup screen there is nothing to add to.
    fn add_btn_rect(&self) -> [f32; 4] {
        self.btn_dock().1[0]
    }

    /// The top-left "PulseMap" wordmark, which doubles as a Home button once a
    /// map is loaded. Matches the label's draw position exactly.
    fn title_rect(&self) -> [f32; 4] {
        let s = self.ui_scale;
        [16.0 * s, 12.0 * s, self.title_w, TITLE_PX * 1.3 * s]
    }

    fn title_hit(&self, cursor: (f64, f64)) -> bool {
        !self.on_startup_screen() && rect_hit(self.title_rect(), cursor)
    }

    /// Back to the empty startup screen. Corrections stay on disk — they key off
    /// file paths, so they still apply when a folder is loaded again.
    fn go_home(&mut self) {
        self.nodes.clear();
        self.sections.clear();
        self.status = Status::Idle;
        self.search_query.clear();
        self.num_instances = 0;
        self.hovered = None;
        self.hovered_legend = None;
        self.active_section = None;
        self.playing = None;
        self.dragging_node = None;
        self.file_drag_arm = None;
        self.drag_target_section = None;
        self.origins.clear();
        self.intro_start = None;
        self.camera = Camera { center: [0.0, 0.0], zoom: 1.2 }; // matches State::new
    }

    fn add_btn_hit(&self, cursor: (f64, f64)) -> bool {
        !self.on_startup_screen() && rect_hit(self.add_btn_rect(), cursor)
    }

    fn legend_hit(&self, cursor: (f64, f64)) -> Option<usize> {
        let rows = hud::legend_layout(&self.sections, self.viewport().0, self.ui_scale);
        rows.iter().position(|r| {
            let [x, y, w, h] = r.rect;
            cursor.0 as f32 >= x && cursor.0 as f32 <= x + w && cursor.1 as f32 >= y && cursor.1 as f32 <= y + h
        })
    }

    /// Snap the camera to frame a section (legend row click).
    fn fly_to_section(&mut self, idx: usize) {
        let Some(s) = self.sections.get(idx) else { return };
        let name = s.name.clone();
        let (cx, cy) = (s.cx, s.cy);

        // Frame the section's ACTUAL node bounds — an absolute fit, not a
        // relative `zoom *= 2.6`. Relative zoom compounded on every repeat click
        // (clicking twice flew straight past the section) and was far too weak
        // to reach a section when starting from fully zoomed out.
        let mut b: Option<[f32; 4]> = None;
        for n in self.nodes.iter().filter(|n| n.section == name) {
            let (x, y) = (n.x, -n.y);
            b = Some(match b {
                None => [x, x, y, y],
                Some([ix, ax, iy, ay]) => [ix.min(x), ax.max(x), iy.min(y), ay.max(y)],
            });
        }
        match b {
            Some([minx, maxx, miny, maxy]) => {
                self.camera.center = [(minx + maxx) * 0.5, (miny + maxy) * 0.5];
                let (vw, vh) = self.viewport();
                let spanx = (maxx - minx).max(NODE_R * 4.0);
                let spany = (maxy - miny).max(NODE_R * 4.0);
                self.camera.zoom =
                    (vw * 0.7 / spanx).min(vh * 0.7 / spany).clamp(0.05, 200.0);
            }
            // Section with no nodes on screen yet (mid-stream): just centre it.
            None => self.camera.center = [cx, -cy],
        }
    }

    /// World position -> physical-px screen position, the inverse of
    /// `Camera::world_at`. Used to place drag-target UI over a section.
    fn world_to_screen(&self, world: [f32; 2]) -> [f32; 2] {
        let (vw, vh) = self.viewport();
        let off = [(world[0] - self.camera.center[0]) * self.camera.zoom, (world[1] - self.camera.center[1]) * self.camera.zoom];
        [vw * 0.5 + off[0], vh * 0.5 - off[1]]
    }

    /// Nearest section (by center distance) to a world position, excluding
    /// one by name (the node's current section) — mirrors the web app's
    /// `nearestSection`. Returns (section index, distance).
    fn nearest_section_to(&self, world: [f32; 2], exclude: &str) -> Option<(usize, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, s) in self.sections.iter().enumerate() {
            if s.name == exclude {
                continue;
            }
            let d = ((world[0] - s.cx).powi(2) + (world[1] - (-s.cy)).powi(2)).sqrt();
            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((i, d));
            }
        }
        best
    }

    /// Move a just-dropped node to where it actually belongs inside its
    /// section: beside its nearest neighbour in CLAP space, wearing that
    /// neighbour's gradient coordinates. Without this the node keeps the spot
    /// the cursor happened to release on and the `rel_x/rel_y` shade from its
    /// *old* section, so it reads as the wrong colour among its new siblings.
    ///
    /// Embeddings are L2-normalized, so the dot product is the cosine.
    /// No cached embedding (or no sibling) leaves the drop position standing.
    fn resettle(&mut self, idx: usize, embs: &std::collections::HashMap<&str, &[f32]>) {
        let Some(me) = embs.get(self.nodes[idx].path.as_str()).copied() else { return };
        let sec = self.nodes[idx].section.clone();
        let peer = self
            .nodes
            .iter()
            .enumerate()
            .filter(|&(i, n)| i != idx && n.section == sec)
            .filter_map(|(i, n)| {
                let e = embs.get(n.path.as_str())?;
                Some((i, me.iter().zip(e.iter()).map(|(a, b)| a * b).sum::<f32>()))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i);
        if let Some(p) = peer {
            // Offset off the peer rather than onto it: an exact overlap gives
            // the separation force no direction to push in. The sim takes it
            // from here into whichever adjacent slot is free.
            self.nodes[idx].home_x = self.nodes[p].home_x + NODE_R;
            self.nodes[idx].home_y = self.nodes[p].home_y + NODE_R;
            self.nodes[idx].rel_x = self.nodes[p].rel_x;
            self.nodes[idx].rel_y = self.nodes[p].rel_y;
        }
    }

    /// Reheat the force sim so the map re-settles (after a drag, as in the web
    /// app's `reheat()`).
    fn reheat(&mut self) {
        self.sim_alpha = 1.0;
    }

    /// Advance the force sim one frame. Returns true while it's still running.
    fn tick_physics(&mut self) -> bool {
        // While analysing the blob is static — nodes only change colour, so
        // there's nothing to simulate and no reason to burn frames (or steal
        // cores from the analysis workers). Sorting starts once Done lands.
        if matches!(self.status, Status::Analyzing { .. }) {
            return false;
        }
        if self.sim_alpha <= SIM_ALPHA_MIN {
            return false;
        }
        let decay = if self.nodes.len() > 800 { 0.006 } else { 0.018 };
        self.sim_alpha += (0.0 - self.sim_alpha) * decay;
        let fixed = self.dragging_node;
        pulsemap::layout::physics_step(&mut self.nodes, self.sim_alpha, fixed);
        if self.sim_alpha <= SIM_ALPHA_MIN {
            self.sim_alpha = 0.0;
        }
        true
    }

    fn show_toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), Instant::now()));
    }

    fn save_corrections(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.corrections) {
            if let Some(dir) = corrections_path().parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(corrections_path(), json);
        }
    }

    /// Audition a node's sound: stop whatever's playing, decode + play it, and
    /// start the waveform playhead for that node.
    fn audition(&mut self, idx: usize) {
        let Some(dev) = self.audio.as_ref() else { return };
        let Some(path) = self.nodes.get(idx).map(|n| n.path.clone()) else { return };
        if let Some(p) = self.player.take() {
            p.stop();
        }
        if let Ok(file) = File::open(&path) {
            if let Ok(dec) = rodio::Decoder::try_from(file) {
                let player = rodio::Player::connect_new(dev.mixer());
                player.append(dec);
                self.player = Some(player);
                self.playing = Some((idx, Instant::now()));
            }
        }
    }

    fn viewport(&self) -> (f32, f32) {
        (self.config.width as f32, self.config.height as f32)
    }

    fn resize(&mut self, new: winit::dpi::PhysicalSize<u32>) {
        if new.width > 0 && new.height > 0 {
            self.size = new;
            self.config.width = new.width;
            self.config.height = new.height;
            self.surface.configure(&self.device, &self.config);
            self.hud.resize(&self.queue, new.width, new.height);
        }
    }

    /// Replace the point cloud with laid-out nodes and fit the camera to them.
    /// World Y is flipped (layout is y-down; camera is y-up).
    /// Grow the instance buffer geometrically — streaming appends one node at a
    /// time and reallocating per sound would be pointless churn.
    fn ensure_instance_capacity(&mut self, n: usize) {
        if n <= self.instance_cap {
            return;
        }
        let cap = n.next_power_of_two().max(256);
        self.instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: (std::mem::size_of::<PointInstance>() * cap) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_cap = cap;
    }

    /// Analysis is starting: fill the centre with `total` grey, unclassified
    /// nodes packed in a sunflower disc. They don't move during analysis —
    /// each just turns its category colour as it's classified.
    fn begin_blob(&mut self, total: usize) {
        let min_d = NODE_R * 2.0 + 1.0;
        let radius = 0.62 * (total.max(1) as f32).sqrt() * min_d;
        self.nodes = (0..total)
            .map(|i| {
                // Sunflower packing: even density, no clumping at the centre.
                let a = i as f32 * 2.399_963;
                let r = radius * ((i as f32 + 0.5) / total as f32).sqrt();
                let (x, y) = (r * a.cos(), r * a.sin());
                Node {
                    path: String::new(),
                    filename: String::new(),
                    section: String::new(),
                    confidence: 0.0,
                    x,
                    y,
                    color: GREY,
                    rel_x: 0.5,
                    rel_y: 0.5,
                    section_id: u32::MAX, // matches no section: no highlight/dim
                    envelope: Vec::new(),
                    duration: 0.0,
                    home_x: x,
                    home_y: y,
                    vx: 0.0,
                    vy: 0.0,
                }
            })
            .collect();
        self.intro_from = vec![[0.0, 0.0]; total];
        self.sections.clear();
        self.colored = 0;
        self.hovered = None;
        self.active_section = None;
        self.num_instances = total as u32;
        self.ensure_instance_capacity(total);
        self.fit_camera();
        self.write_instances(1.0);
    }

    /// One sound came back classified: colour the next grey node in the blob.
    fn color_next(&mut self, filename: String, section: String, confidence: f32) {
        use pulsemap::layout::{color_for, section_id};
        let Some(n) = self.nodes.get_mut(self.colored) else { return };
        let color = color_for(&section);
        n.filename = filename;
        n.confidence = confidence;
        n.color = color;
        n.section_id = section_id(&section);
        n.section = section.clone();
        self.colored += 1;

        match self.sections.iter_mut().find(|s| s.name == section) {
            Some(s) => s.count += 1,
            None => self.sections.push(SectionMeta {
                name: section,
                count: 1,
                color,
                cx: 0.0,
                cy: 0.0,
            }),
        }
    }

    /// Analysis finished: swap in the real layout but keep every node where it
    /// currently sits, so the sim glides them into their final spots instead of
    /// the map snapping/replaying an intro over an already-populated screen.
    fn morph_to(&mut self, nodes: &[Node], sections: Vec<SectionMeta>) {
        let prev: std::collections::HashMap<String, (f32, f32)> =
            self.nodes.iter().map(|n| (n.filename.clone(), (n.x, n.y))).collect();
        self.nodes = nodes.to_vec();
        for n in &mut self.nodes {
            if let Some(&(x, y)) = prev.get(&n.filename) {
                n.x = x;
                n.y = y;
            }
        }
        self.intro_from = vec![[0.0, 0.0]; self.nodes.len()];
        self.sections = sections;
        self.hovered = None;
        self.active_section = None;
        self.hovered_legend = None;
        self.num_instances = self.nodes.len() as u32;
        self.ensure_instance_capacity(self.nodes.len());
        self.fit_camera();
        self.reheat();
        self.write_instances(1.0);
    }

    fn set_nodes(&mut self, nodes: &[Node], sections: Vec<SectionMeta>) {
        self.nodes = nodes.to_vec();
        self.sections = sections;
        self.hovered = None;
        self.active_section = None;
        self.hovered_legend = None;
        self.num_instances = nodes.len() as u32;
        if nodes.is_empty() {
            return;
        }
        self.ensure_instance_capacity(nodes.len());
        self.fit_camera();

        // Intro: every node starts grey at the map center, flies out to its
        // home position and colorizes on arrival (mirrors app.js's reveal).
        self.intro_from = vec![self.camera.center; nodes.len()];
        self.intro_start = Some(Instant::now());
        // Hand off to physics for the organic settle once the map lands.
        self.reheat();
        self.write_instances(0.0);
    }

    /// Frame the camera to the current node bounds.
    fn fit_camera(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let (mut minx, mut maxx, mut miny, mut maxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for n in &self.nodes {
            let (x, y) = (n.x, -n.y);
            minx = minx.min(x);
            maxx = maxx.max(x);
            miny = miny.min(y);
            maxy = maxy.max(y);
        }
        self.camera.center = [(minx + maxx) * 0.5, (miny + maxy) * 0.5];
        let (vw, vh) = self.viewport();
        let spanx = (maxx - minx).max(1.0);
        let spany = (maxy - miny).max(1.0);
        self.camera.zoom = (vw * 0.85 / spanx).min(vh * 0.85 / spany).clamp(0.05, 200.0);
    }

    /// Rebuild the instance buffer at intro progress `t` (0=start scatter/grey,
    /// 1=home/final color). Ease with smoothstep, like a CSS transition.
    fn write_instances(&mut self, t: f32) {
        let e = t * t * (3.0 - 2.0 * t); // smoothstep easing
        let insts: Vec<PointInstance> = self
            .nodes
            .iter()
            .zip(&self.intro_from)
            .map(|(n, from)| {
                let home = [n.x, -n.y];
                PointInstance {
                    pos: [lerp(from[0], home[0], e), lerp(from[1], home[1], e)],
                    color: [
                        lerp(GREY[0], n.color[0], e),
                        lerp(GREY[1], n.color[1], e),
                        lerp(GREY[2], n.color[2], e),
                    ],
                    confidence: n.confidence,
                    rel: [n.rel_x, n.rel_y],
                    section_id: n.section_id,
                    dim: self.search_dim(n),
                }
            })
            .collect();
        self.queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&insts));
    }

    /// Advance the intro animation. Returns true while still animating.
    fn tick_intro(&mut self) -> bool {
        let Some(start) = self.intro_start else { return false };
        let t = (start.elapsed().as_secs_f32() / INTRO_DURATION.as_secs_f32()).min(1.0);
        self.write_instances(t);
        if t >= 1.0 {
            self.intro_start = None;
            false
        } else {
            true
        }
    }

    /// Current intro-animation progress (1.0 once settled). Used to re-write
    /// the instance buffer (e.g. after a search keystroke) without skipping
    /// an animation still in flight.
    fn current_intro_t(&self) -> f32 {
        match self.intro_start {
            Some(start) => (start.elapsed().as_secs_f32() / INTRO_DURATION.as_secs_f32()).min(1.0),
            None => 1.0,
        }
    }

    /// Visual weight for search filtering: full weight if the query is empty
    /// or the node matches (filename/section, case-insensitive substring);
    /// heavily dimmed otherwise — mirrors the web app's search-miss dimming.
    fn search_dim(&self, n: &Node) -> f32 {
        if self.search_query.is_empty() {
            return 1.0;
        }
        let q = self.search_query.to_lowercase();
        if n.filename.to_lowercase().contains(&q) || n.section.to_lowercase().contains(&q) {
            1.0
        } else {
            0.12
        }
    }

    fn upload_camera(&self) {
        let u = CameraUniform {
            center: self.camera.center,
            viewport: self.viewport().into(),
            zoom: self.camera.zoom,
            point_px: NODE_R,
            hovered: self.hovered.map(|i| i as f32).unwrap_or(-1.0),
            hovered_section: self.active_section.map(|s| s as f32).unwrap_or(-1.0),
            time: self.launched.elapsed().as_secs_f32(),
            _pad: [0.0; 3],
        };
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&u));
    }

    /// Build this frame's HUD content: title, progress/ETA panel, hover
    /// banner, and section legend. Pure/read-only — owned Vecs, no borrows of
    /// `self` survive past this call, so it composes cleanly with `hud.draw`
    /// (which needs `&mut self.hud` while `self.device`/`self.queue` are
    /// borrowed immutably — disjoint fields, no conflict).
    fn build_hud_content(&self) -> (Vec<hud::Rect>, Vec<hud::Label>) {
        let (vw, vh) = self.viewport();
        let s = self.ui_scale; // physical-px surface; every HUD size/pos scales by DPI
        let mut rects = Vec::new();
        let mut labels = Vec::new();

        // Bottom dock of actions, or the startup screen's own layout. On the
        // dock the panel carries the background and each button is just a label
        // that lights up on hover — three separate grey pills read as clutter.
        if self.on_startup_screen() {
            // Primary action on the empty screen — big and accent-filled.
            let [bx, by, bw, bh] = self.open_btn_rect();
            let hover = self.open_btn_hit(self.cursor);
            let tsize = 15.0 * s;
            rects.push(hud::Rect {
                pos: [bx, by],
                size: [bw, bh],
                radius: bh * 0.5,
                color: if hover { [0.36, 0.51, 0.93, 1.0] } else { [0.29, 0.42, 0.80, 1.0] },
            });
            labels.push(hud::Label {
                text: OPEN_BTN_LABEL.to_string(),
                pos: [bx + (bw - self.open_btn_w) * 0.5, by + (bh - tsize * 1.3) * 0.5],
                size_px: tsize,
                color: hud::COL_TEXT,
                bold: false,
                max_width: bw,
            });
            // Clear-cache keeps its own corner here; there is no dock yet.
            let [cx, cy, cw, ch] = self.clear_btn_rect();
            let chover = self.clear_btn_hit(self.cursor);
            let ts = BTN_TEXT_PX * s;
            rects.push(hud::Rect {
                pos: [cx, cy],
                size: [cw, ch],
                radius: ch * 0.5,
                color: if chover { [0.1647, 0.1843, 0.2275, 1.0] } else { hud::COL_PANEL },
            });
            labels.push(hud::Label {
                text: CLEAR_BTN_LABEL.to_string(),
                pos: [cx + (cw - self.clear_btn_w) * 0.5, cy + (ch - ts * 1.3) * 0.5],
                size_px: ts,
                color: if chover { hud::COL_TEXT } else { hud::COL_DIM },
                bold: false,
                max_width: cw,
            });
        } else {
            let (panel, items) = self.btn_dock();
            rects.push(hud::Rect {
                pos: [panel[0], panel[1]],
                size: [panel[2], panel[3]],
                radius: 10.0 * s,
                color: hud::COL_PANEL,
            });
            let tsize = BTN_TEXT_PX * s;
            let btns = [(ADD_BTN_LABEL, self.add_btn_w), (CLEAR_BTN_LABEL, self.clear_btn_w)];
            for (i, (label, lw)) in btns.iter().enumerate() {
                let [bx, by, bw, bh] = items[i];
                let hover = rect_hit(items[i], self.cursor);
                if hover {
                    rects.push(hud::Rect {
                        pos: [bx, by],
                        size: [bw, bh],
                        radius: bh * 0.5,
                        color: [0.1647, 0.1843, 0.2275, 1.0],
                    });
                }
                labels.push(hud::Label {
                    text: label.to_string(),
                    pos: [bx + (bw - lw) * 0.5, by + (bh - tsize * 1.3) * 0.5],
                    size_px: tsize,
                    color: if hover { hud::COL_TEXT } else { hud::COL_DIM },
                    bold: false,
                    max_width: bw,
                });
            }
        }

        // Empty/startup state: nothing loaded and not analyzing yet — big
        // centered title + instructions, matching the web app's dropzone.
        if self.nodes.is_empty() && matches!(self.status, Status::Idle) {
            let title = "PulseMap";
            let title_size = 40.0 * s;
            let title_w = title.len() as f32 * title_size * 0.62; // rough centering estimate
            labels.push(hud::Label {
                text: title.to_string(),
                pos: [(vw - title_w) * 0.5, vh * 0.5 - 70.0 * s],
                size_px: title_size,
                color: hud::COL_TEXT,
                bold: true,
                max_width: vw,
            });
            let sub = "Drop a folder of samples anywhere in this window, or:";
            let sub_size = 16.0 * s;
            let sub_w = sub.len() as f32 * sub_size * 0.48;
            labels.push(hud::Label {
                text: sub.to_string(),
                pos: [(vw - sub_w) * 0.5, vh * 0.5 - 8.0 * s],
                size_px: sub_size,
                color: hud::COL_DIM,
                bold: false,
                max_width: vw,
            });
            let note = "Everything is analyzed locally — nothing leaves your machine";
            let note_size = 13.0 * s;
            let note_w = note.len() as f32 * note_size * 0.46;
            labels.push(hud::Label {
                text: note.to_string(),
                pos: [(vw - note_w) * 0.5, vh * 0.5 + 22.0 * s],
                size_px: note_size,
                color: hud::COL_DIM,
                bold: false,
                max_width: vw,
            });
            return (rects, labels);
        }

        // Title, top-left (once loaded / while working). y is set so the title's
        // optical centre matches the search box's (both centre on 25*s).
        labels.push(hud::Label {
            text: TITLE.to_string(),
            pos: [16.0 * s, 12.0 * s],
            size_px: TITLE_PX * s,
            // Brightens on hover — the only hint that the wordmark is a Home button.
            color: if self.title_hit(self.cursor) { hud::COL_DIM } else { hud::COL_TEXT },
            bold: true,
            max_width: 300.0 * s,
        });

        // Search box, top area next to the title — mirrors the web app's
        // #topbar search input. Always "focused" (this app has no other text
        // field), filters nodes by filename/section (see `search_dim`).
        {
            let box_w = 260.0 * s;
            let box_h = 30.0 * s;
            let box_x = 130.0 * s;
            let box_y = 10.0 * s;
            let active = !self.search_query.is_empty();
            rects.push(hud::Rect {
                pos: [box_x, box_y],
                size: [box_w, box_h],
                radius: 8.0 * s,
                color: if active { [0.1647, 0.1843, 0.2275, 1.0] } else { hud::COL_PANEL },
            });
            let (text, color) = if self.search_query.is_empty() {
                ("Start typing to search sounds".to_string(), hud::COL_DIM)
            } else {
                (self.search_query.clone(), hud::COL_TEXT)
            };
            labels.push(hud::Label {
                text,
                pos: [box_x + 12.0 * s, box_y + 7.0 * s],
                size_px: 13.0 * s,
                color,
                bold: false,
                max_width: box_w - 24.0 * s,
            });
        }

        // Progress panel, centered, while analyzing.
        if let Status::Analyzing { done, total, eta_secs } = &self.status {
            let panel_w = 360.0 * s;
            let panel_h = 110.0 * s;
            let px = (vw - panel_w) * 0.5;
            let py = 90.0 * s;
            rects.push(hud::Rect { pos: [px, py], size: [panel_w, panel_h], radius: 12.0 * s, color: hud::COL_PANEL });
            let label = if *total > 0 {
                format!("Analyzing {done}/{total} sounds…")
            } else {
                "Getting ready…".to_string()
            };
            labels.push(hud::Label {
                text: label,
                pos: [px + 20.0 * s, py + 20.0 * s],
                size_px: 18.0 * s,
                color: hud::COL_TEXT,
                bold: false,
                max_width: panel_w - 40.0 * s,
            });
            let bar_w = panel_w - 40.0 * s;
            let bar_y = py + 56.0 * s;
            rects.push(hud::Rect {
                pos: [px + 20.0 * s, bar_y],
                size: [bar_w, 6.0 * s],
                radius: 3.0 * s,
                color: [0.125, 0.141, 0.176, 1.0], // track
            });
            if *total > 0 {
                let frac = (*done as f32 / *total as f32).clamp(0.0, 1.0);
                rects.push(hud::Rect {
                    pos: [px + 20.0 * s, bar_y],
                    size: [bar_w * frac, 6.0 * s],
                    radius: 3.0 * s,
                    color: hud::COL_ACCENT,
                });
            }
            let tail = match eta_secs {
                Some(sec) => format!("~{sec}s left"),
                None => String::new(),
            };
            labels.push(hud::Label {
                text: tail,
                pos: [px + 20.0 * s, bar_y + 16.0 * s],
                size_px: 14.0 * s,
                color: hud::COL_DIM,
                bold: false,
                max_width: bar_w,
            });
        }

        // Info dock, bottom-right: the hovered/playing sound's name on the left
        // and its waveform on the right, in one panel. The waveform is always
        // present — a flat line at rest, rising into THIS sound's own peak-
        // amplitude envelope when it plays (a kick's sharp spike looks nothing
        // like an open hat's long decay), then settling back to flat.
        if !self.nodes.is_empty() {
            let [dx, dy, dw, dh] = self.info_dock();
            rects.push(hud::Rect {
                pos: [dx, dy],
                size: [dw, dh],
                radius: 10.0 * s,
                color: hud::COL_PANEL,
            });

            let text = match self.hovered.map(|i| &self.nodes[i]) {
                Some(n) => {
                    format!("{} · {} {}%", n.filename, n.section, (n.confidence * 100.0).round() as i32)
                }
                None => "Hover a sound to preview".to_string(),
            };
            let tsize = 14.0 * s;
            let text_w = dw - WAVE_W * s - 34.0 * s; // label column, waveform gets the rest
            labels.push(hud::Label {
                text,
                pos: [dx + 14.0 * s, dy + (dh - tsize * 1.3) * 0.5],
                size_px: tsize,
                color: if self.hovered.is_some() { hud::COL_TEXT } else { hud::COL_DIM },
                bold: false,
                max_width: text_w,
            });

            let w = WAVE_W * s;
            let h = 22.0 * s;
            let x0 = dx + dw - w - 14.0 * s;
            let y0 = dy + (dh - h) * 0.5;

            // Envelope + attack/release so it animates flat -> shape -> flat.
            let play = self.playing.and_then(|(i, t)| self.nodes.get(i).map(|n| (n, t)));
            let (anim, prog, col) = match play {
                Some((n, started)) => {
                    let e = started.elapsed().as_secs_f32();
                    let dur = n.duration.max(0.05);
                    let anim = if e < WAVE_ATTACK {
                        e / WAVE_ATTACK
                    } else if e <= dur {
                        1.0
                    } else {
                        // Smoothstep rather than linear: eases out of full shape
                        // and into flat, so neither end of the settle snaps.
                        let x = ((e - dur) / WAVE_RELEASE).clamp(0.0, 1.0);
                        1.0 - x * x * (3.0 - 2.0 * x)
                    };
                    (anim, (e / dur).clamp(0.0, 1.0), n.color)
                }
                None => (0.0, 0.0, [0.45, 0.48, 0.56]),
            };

            let bw = w / WAVE_BARS as f32;
            for i in 0..WAVE_BARS {
                let target = play.and_then(|(n, _)| n.envelope.get(i).copied()).unwrap_or(0.0);
                let bh = (target * anim * h).max(2.0 * s); // 2px floor == the flat line
                let a = match play {
                    // Played bars are lit; the rest sit back until the playhead
                    // reaches them.
                    Some(_) if (i as f32 + 0.5) / WAVE_BARS as f32 <= prog => 0.95,
                    Some(_) => 0.30,
                    None => 0.40,
                };
                rects.push(hud::Rect {
                    pos: [x0 + i as f32 * bw, y0 + (h - bh) * 0.5],
                    size: [(bw - 1.5 * s).max(1.0 * s), bh],
                    radius: 1.0 * s,
                    color: [col[0], col[1], col[2], a],
                });
            }
        }

        // Section legend, top-right.
        if !self.sections.is_empty() {
            let rows = hud::legend_layout(&self.sections, vw, s);
            let panel_top = 16.0 * s;
            let panel_bottom = rows.last().map(|r| r.rect[1] + r.rect[3]).unwrap_or(50.0 * s) + 8.0 * s;
            let panel_x = rows.first().map(|r| r.rect[0] - 10.0 * s).unwrap_or(vw - 236.0 * s);
            rects.push(hud::Rect {
                pos: [panel_x, panel_top],
                size: [230.0 * s, panel_bottom - panel_top],
                radius: 10.0 * s,
                color: hud::COL_PANEL,
            });
            labels.push(hud::Label {
                text: "Sections".to_string(),
                pos: [panel_x + 12.0 * s, panel_top + 10.0 * s],
                size_px: 14.0 * s,
                color: hud::COL_DIM,
                bold: true,
                max_width: 200.0 * s,
            });
            for (i, row) in rows.iter().enumerate() {
                let is_active = self.hovered_legend == Some(i)
                    || self.active_section == Some(pulsemap::layout::section_id(&row.section));
                if is_active {
                    rects.push(hud::Rect {
                        pos: [row.rect[0], row.rect[1]],
                        size: [row.rect[2], row.rect[3]],
                        radius: 6.0 * s,
                        color: [1.0, 1.0, 1.0, 0.06],
                    });
                }
                rects.push(hud::Rect { pos: [row.swatch[0], row.swatch[1]], size: [row.swatch[2], row.swatch[3]], radius: 3.0 * s, color: row.color });
                labels.push(hud::Label {
                    text: row.section.clone(),
                    pos: [row.rect[0] + 28.0 * s, row.rect[1] + 6.0 * s],
                    size_px: 15.0 * s,
                    color: hud::COL_TEXT,
                    bold: false,
                    max_width: row.rect[2] - 80.0 * s,
                });
                labels.push(hud::Label {
                    text: row.count.to_string(),
                    pos: [row.rect[0] + row.rect[2] - 36.0 * s, row.rect[1] + 6.0 * s],
                    size_px: 15.0 * s,
                    color: hud::COL_DIM,
                    bold: false,
                    max_width: 30.0 * s,
                });
            }
        }

        // Drag-target ring: while dragging a node near a section, show where
        // it'll drop (mirrors the web app's dashed ring).
        if let Some(sec_idx) = self.drag_target_section {
            if let Some(sec) = self.sections.get(sec_idx) {
                let center = self.world_to_screen([sec.cx, -sec.cy]);
                let outer = 34.0 * s;
                let inner = 26.0 * s;
                rects.push(hud::Rect {
                    pos: [center[0] - outer / 2.0, center[1] - outer / 2.0],
                    size: [outer, outer],
                    radius: outer / 2.0,
                    color: hud::COL_ACCENT,
                });
                rects.push(hud::Rect {
                    pos: [center[0] - inner / 2.0, center[1] - inner / 2.0],
                    size: [inner, inner],
                    radius: inner / 2.0,
                    color: [0.05490, 0.05882, 0.07451, 1.0], // punch through to bg -> ring look
                });
            }
        }

        // Toast — auto-expires (see AboutToWait). Stacks directly above the info
        // dock and shares its right edge, rather than landing on top of it.
        if let Some((msg, started)) = &self.toast {
            if started.elapsed() < TOAST_DURATION {
                let text_size = 13.0 * s;
                let w = 20.0 * s + msg.len() as f32 * text_size * 0.55;
                let h = 36.0 * s;
                let x = vw - w - DOCK_PAD * s;
                let y = self.info_dock()[1] - h - 8.0 * s;
                rects.push(hud::Rect { pos: [x, y], size: [w, h], radius: 8.0 * s, color: hud::COL_PANEL });
                labels.push(hud::Label {
                    text: msg.clone(),
                    pos: [x + 14.0 * s, y + 12.0 * s],
                    size_px: text_size,
                    color: hud::COL_TEXT,
                    bold: false,
                    max_width: w - 28.0 * s,
                });
            }
        }

        (rects, labels)
    }

    fn render(&mut self) -> RenderOutcome {
        self.upload_camera();
        let (surface_texture, suboptimal) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => (t, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => (t, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return RenderOutcome::Skip
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                return RenderOutcome::Reconfigure
            }
            wgpu::CurrentSurfaceTexture::Validation => return RenderOutcome::Fatal,
        };
        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        let (rects, labels) = self.build_hud_content();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05490, g: 0.05882, b: 0.07451, a: 1.0 }), // web app's --bg: #0e0f13, exact match
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            if self.num_instances > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_buf.slice(..));
                pass.set_vertex_buffer(1, self.instance_buf.slice(..));
                pass.draw(0..QUAD.len() as u32, 0..self.num_instances);
            }
            self.hud.draw(&self.device, &self.queue, &mut pass, &rects, &labels);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(surface_texture);
        if suboptimal {
            RenderOutcome::Reconfigure
        } else {
            RenderOutcome::Rendered
        }
    }
}

/// What the event loop should do after a render attempt, replacing the old
/// `Result<(), wgpu::SurfaceError>` now that `get_current_texture()` returns a
/// plain enum (`wgpu::CurrentSurfaceTexture`) instead of a `Result`.
enum RenderOutcome {
    Rendered,
    Skip,        // timed out / occluded: try again next frame, nothing wrong
    Reconfigure, // surface out of date: caller should resize/reconfigure
    Fatal,       // validation error: caller should log/exit
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("PulseMap — drop a folder of samples")
            .with_inner_size(winit::dpi::LogicalSize::new(1100.0, 760.0))
            .build(&event_loop)
            .unwrap(),
    );

    let mut state = pollster::block_on(State::new(window.clone()));
    let mut rx: Option<Receiver<Msg>> = None;
    let mut analyzing = false;
    // Folder set queued by a drop or the picker; started on the next AboutToWait.
    // Always the COMPLETE set to map — "Add folder" queues old + new, so the
    // kickoff path below stays a single "analyze these folders" call.
    let mut pending_folders: Option<Vec<PathBuf>> = None;
    // Folders currently on the map, so a later add can re-run the whole set.
    let mut mapped_folders: Vec<PathBuf> = Vec::new();
    let mut analysis_start: Option<Instant> = None;

    // Preload the model: workers compile it now, while the window is up and the
    // user is picking a folder, so the first drop skips the ~1.3s stall.
    // Benchmarked sweet spot: more workers than this made throughput WORSE
    // (8x1 = 41.7 ms/file vs 4x2 = 33.7). See Analyzer::new for the numbers.
    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(pulsemap::analyzer::BEST_WORKERS);
    let assets = asset_dir();
    let pool = AnalyzerPool::new(num_workers, &assets, &assets.join("model.json"));

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::ModifiersChanged(mods) => {
                        // Cmd on macOS, Ctrl everywhere else — Super+drag is an OS
                        // window gesture on Windows and most Linux WMs, so it would
                        // never reach us there.
                        state.cmd_down = if cfg!(target_os = "macos") {
                            mods.state().super_key()
                        } else {
                            mods.state().control_key()
                        };
                    }

                    WindowEvent::KeyboardInput { event: key, .. } => {
                        if key.state != ElementState::Pressed {
                            // ignore releases; ignore key-repeat only for Escape (repeat
                            // shouldn't rapid-fire quit/clear)
                        } else if key.physical_key == PhysicalKey::Code(KeyCode::Escape) && !key.repeat {
                            if state.search_query.is_empty() {
                                elwt.exit();
                            } else {
                                state.search_query.clear();
                                state.write_instances(state.current_intro_t());
                                window.request_redraw();
                            }
                        } else if key.physical_key == PhysicalKey::Code(KeyCode::Backspace) {
                            if state.search_query.pop().is_some() {
                                state.write_instances(state.current_intro_t());
                                window.request_redraw();
                            }
                        } else if let Some(text) = &key.text {
                            // Printable characters only — modifier/function keys carry
                            // no `text`, and this filters out stray control chars.
                            let printable: String = text.chars().filter(|c| !c.is_control()).collect();
                            if !printable.is_empty() {
                                state.search_query.push_str(&printable);
                                state.write_instances(state.current_intro_t());
                                window.request_redraw();
                            }
                        }
                    }

                    WindowEvent::Resized(new) => {
                        state.resize(new);
                        window.request_redraw();
                    }

                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        state.ui_scale = scale_factor as f32; // moved to a different-DPI display
                        state.clear_btn_w =
                            state.hud.measure(CLEAR_BTN_LABEL, BTN_TEXT_PX * state.ui_scale, false);
                        state.add_btn_w =
                            state.hud.measure(ADD_BTN_LABEL, BTN_TEXT_PX * state.ui_scale, false);
                        state.open_btn_w =
                            state.hud.measure(OPEN_BTN_LABEL, 15.0 * state.ui_scale, false);
                        state.hud.clear_text_cache(); // every cached size is now stale
                        window.request_redraw();
                    }

                    WindowEvent::DroppedFile(path) => {
                        // Drop-to-load is startup-screen only. Once a map is up the
                        // Choose-folder button is the only way in: a sample dragged
                        // out to a DAW and fumbled back over the window would
                        // otherwise re-analyze that one file and wipe the map.
                        if !analyzing && state.on_startup_screen() {
                            pending_folders = Some(vec![path]);
                        }
                    }

                    WindowEvent::MouseInput { button: MouseButton::Left, state: st, .. } => {
                        if st == ElementState::Pressed {
                            if state.open_btn_hit(state.cursor) {
                                // Native folder picker. Blocking + main-thread is
                                // required on macOS (AppKit modals can't run off
                                // the main thread), and it's a user-initiated
                                // modal so stalling the loop here is expected.
                                if !analyzing {
                                    if let Some(dir) = rfd::FileDialog::new()
                                        .set_title("Choose a folder of samples")
                                        .pick_folder()
                                    {
                                        pending_folders = Some(vec![dir]);
                                    }
                                }
                                state.panning = false;
                                window.request_redraw();
                            } else if state.title_hit(state.cursor) {
                                // Wordmark doubles as Home: drop the map and go back
                                // to the startup screen.
                                if !analyzing {
                                    state.go_home();
                                    mapped_folders.clear();
                                }
                                state.panning = false;
                                window.request_redraw();
                            } else if state.add_btn_hit(state.cursor) {
                                // Same blocking main-thread picker as Choose-folder,
                                // but queues the existing folders alongside the new
                                // one so the map grows instead of being replaced.
                                if !analyzing {
                                    if let Some(dir) = rfd::FileDialog::new()
                                        .set_title("Add a folder of samples")
                                        .pick_folder()
                                    {
                                        let mut all = mapped_folders.clone();
                                        all.push(dir);
                                        pending_folders = Some(all);
                                    }
                                }
                                state.panning = false;
                                window.request_redraw();
                            } else if state.clear_btn_hit(state.cursor) {
                                match pool.cache.lock().unwrap().clear() {
                                    Ok(n) => state.show_toast(format!("✓ cleared {n} cached sounds")),
                                    Err(e) => state.show_toast(format!("✗ clear failed: {e}")),
                                }
                                state.panning = false;
                                window.request_redraw();
                            } else if let Some(row) = state.legend_hit(state.cursor) {
                                state.fly_to_section(row); // click a legend row -> fly there
                                state.panning = false;
                                window.request_redraw();
                            } else if let Some(idx) = state.node_hit(state.cursor) {
                                if state.cmd_down {
                                    // Cmd-drag grabs the node itself: move it, and
                                    // reclassify if it lands in another section.
                                    state.dragging_node = Some(idx);
                                    // First move only: keep the ORIGINAL spot, so
                                    // a there-and-back drag is a true round trip.
                                    let n = &state.nodes[idx];
                                    state.origins.entry(n.path.clone()).or_insert((
                                        n.section.clone(),
                                        n.home_x,
                                        n.home_y,
                                        n.rel_x,
                                        n.rel_y,
                                    ));
                                } else {
                                    // Plain drag belongs to the OS — hand the file
                                    // out so it can be dropped into a DAW.
                                    state.file_drag_arm = Some((idx, state.cursor));
                                }
                                state.hovered = Some(idx);
                            } else {
                                state.panning = true;
                            }
                        } else if let Some(idx) = state.dragging_node.take() {
                            let node_world = [state.nodes[idx].x, -state.nodes[idx].y];
                            let cur_section = state.nodes[idx].section.clone();
                            let target = state
                                .nearest_section_to(node_world, &cur_section)
                                .filter(|&(_, d)| d < DROP_RADIUS);
                            if let Some((sec_idx, _)) = target {
                                let new_name = state.sections[sec_idx].name.clone();
                                if let Some(old_idx) = state.sections.iter().position(|s| s.name == cur_section) {
                                    state.sections[old_idx].count = state.sections[old_idx].count.saturating_sub(1);
                                }
                                state.sections[sec_idx].count += 1;
                                state.nodes[idx].color = pulsemap::layout::color_for(&new_name);
                                state.nodes[idx].section_id = pulsemap::layout::section_id(&new_name);
                                state.nodes[idx].section = new_name.clone();
                                let path = state.nodes[idx].path.clone();
                                let came_from = state
                                    .origins
                                    .get(&path)
                                    .filter(|o| o.0 == new_name)
                                    .cloned();
                                if let Some((_, hx, hy, rx, ry)) = came_from {
                                    // Back where it started: restore the exact spot
                                    // and drop the correction, so the map matches
                                    // what it looked like before the first move.
                                    let n = &mut state.nodes[idx];
                                    (n.home_x, n.home_y, n.rel_x, n.rel_y) = (hx, hy, rx, ry);
                                    state.origins.remove(&path);
                                    state.corrections.remove(&path);
                                } else {
                                    // Where you dropped it is only a starting point:
                                    // the node belongs next to its nearest neighbour
                                    // in the new section, not where the cursor let go.
                                    state.nodes[idx].home_x = state.nodes[idx].x;
                                    state.nodes[idx].home_y = state.nodes[idx].y;
                                    let cache = pool.cache.lock().unwrap();
                                    state.resettle(idx, &cache.embeddings_by_path());
                                    state.corrections.insert(path, new_name.clone());
                                }
                                state.save_corrections();
                                state.show_toast(format!("✓ moved to {new_name}"));
                            }
                            // Release to the sim either way: a valid drop settles
                            // into its new section, an invalid one springs back to
                            // its untouched home. Physics handles both.
                            state.drag_target_section = None;
                            state.reheat();
                            state.write_instances(state.current_intro_t());
                            window.request_redraw();
                        } else {
                            state.panning = false;
                            state.file_drag_arm = None; // click without a move: no drag-out
                        }
                    }

                    WindowEvent::CursorMoved { position, .. } => {
                        let new = (position.x, position.y);
                        // Repaint when the cursor crosses the Clear-cache button
                        // boundary so its hover highlight tracks (the highlight is
                        // derived from `cursor` at draw time, not stored state).
                        if state.clear_btn_hit(new) != state.clear_btn_hit(state.cursor) {
                            window.request_redraw();
                        }
                        if let Some((idx, p0)) = state.file_drag_arm {
                            // Past the slop threshold this is a drag, not a click:
                            // hand the file to the OS. Clear our own press state
                            // first — once AppKit owns the drag session the button
                            // release is delivered to it, never back to us.
                            if (new.0 - p0.0).hypot(new.1 - p0.1) > 4.0 {
                                state.file_drag_arm = None;
                                state.panning = false;
                                let path = PathBuf::from(&state.nodes[idx].path);
                                let path = std::fs::canonicalize(&path).unwrap_or(path);
                                if let Err(e) = drag::start_drag(
                                    &*window,
                                    drag::DragItem::Files(vec![path]),
                                    drag::Image::Raw(swatch_bmp(state.nodes[idx].color)),
                                    |_, _| {},
                                    Default::default(),
                                ) {
                                    log::warn!("drag-out failed: {e}");
                                }
                            }
                        } else if let Some(idx) = state.dragging_node {
                            let world = state.camera.world_at(new, state.viewport());
                            state.nodes[idx].x = world[0];
                            state.nodes[idx].y = -world[1];
                            let cur_section = state.nodes[idx].section.clone();
                            state.drag_target_section = state
                                .nearest_section_to(world, &cur_section)
                                .filter(|&(_, d)| d < DROP_RADIUS)
                                .map(|(i, _)| i);
                            // Keep the sim hot while dragging so neighbours give
                            // way as the node is pushed through them.
                            state.reheat();
                            state.write_instances(state.current_intro_t());
                            window.request_redraw();
                        } else if state.panning {
                            let dx = (new.0 - state.cursor.0) as f32;
                            let dy = (new.1 - state.cursor.1) as f32;
                            state.camera.center[0] -= dx / state.camera.zoom;
                            state.camera.center[1] += dy / state.camera.zoom;
                            window.request_redraw();
                        } else if let Some(row) = state.legend_hit(new) {
                            // Hovering the legend highlights that section on the
                            // map too (same gradient+dim as hovering its nodes).
                            if state.hovered_legend != Some(row) {
                                state.hovered_legend = Some(row);
                                state.active_section =
                                    Some(pulsemap::layout::section_id(&state.sections[row].name));
                                state.hovered = None;
                                window.request_redraw();
                            }
                        } else {
                            if state.hovered_legend.is_some() {
                                state.hovered_legend = None;
                                window.request_redraw();
                            }
                            let near = state.nearest_node(new);
                            // Active category: cursor anywhere within a cluster
                            // (between same-category nodes) → highlight it.
                            let active = near
                                .filter(|&(_, d)| d < 8.0)
                                .map(|(i, _)| state.nodes[i].section_id);
                            // Hovered node: must be on the node → audition + grow.
                            let hit_r = NODE_R.max(7.0 / state.camera.zoom);
                            let hov = near.filter(|&(_, d)| d < hit_r).map(|(i, _)| i);

                            if hov != state.hovered {
                                state.hovered = hov;
                                if let Some(i) = hov {
                                    state.audition(i);
                                }
                                window.request_redraw();
                            }
                            if active != state.active_section {
                                state.active_section = active;
                                window.request_redraw();
                            }
                        }
                        state.cursor = new;
                    }

                    WindowEvent::MouseWheel { delta, .. } => {
                        let steps = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 60.0,
                        };
                        let vp = state.viewport();
                        let before = state.camera.world_at(state.cursor, vp);
                        state.camera.zoom = (state.camera.zoom * 1.12_f32.powf(steps)).clamp(0.05, 200.0);
                        let after = state.camera.world_at(state.cursor, vp);
                        state.camera.center[0] += before[0] - after[0];
                        state.camera.center[1] += before[1] - after[1];
                        window.request_redraw();
                    }

                    WindowEvent::RedrawRequested => match state.render() {
                        RenderOutcome::Rendered | RenderOutcome::Skip => {}
                        RenderOutcome::Reconfigure => state.resize(state.size),
                        RenderOutcome::Fatal => log::warn!("surface validation error"),
                    },

                    _ => {}
                },

                Event::AboutToWait => {
                    // Single kickoff path for both drag-drop and the picker.
                    if let Some(folders) = pending_folders.take() {
                        if !analyzing {
                            mapped_folders = folders.clone();
                            analyzing = true;
                            analysis_start = Some(Instant::now());
                            state.status = Status::Analyzing { done: 0, total: 0, eta_secs: None };
                            let (tx, r) = channel();
                            rx = Some(r);
                            pool.analyze_folders(folders, state.corrections.clone(), tx);
                            elwt.set_control_flow(ControlFlow::Poll);
                            window.request_redraw();
                        }
                    }

                    // Drain analysis messages.
                    let mut instances_dirty = false;
                    if let Some(r) = &rx {
                        while let Ok(msg) = r.try_recv() {
                            match msg {
                                Msg::Progress { done, total } => {
                                    let eta = analysis_start.and_then(|s| {
                                        (done > 0 && total > done).then(|| {
                                            let per = s.elapsed().as_secs_f32() / done as f32;
                                            (per * (total - done) as f32).ceil() as u32
                                        })
                                    });
                                    state.status = Status::Analyzing { done, total, eta_secs: eta };
                                    window.request_redraw();
                                }
                                Msg::Start { total } => {
                                    state.begin_blob(total);
                                    window.request_redraw();
                                }
                                Msg::Sorted { filename, section, confidence } => {
                                    state.color_next(filename, section, confidence);
                                    // Batched: the whole drain shares one buffer
                                    // rewrite below, not one per sound.
                                    instances_dirty = true;
                                }
                                Msg::Done(nodes, sections) => {
                                    let took = analysis_start.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);
                                    state.status = Status::Done { count: nodes.len(), took_secs: took };
                                    window.set_title("PulseMap");
                                    // Streamed nodes are already on screen — glide
                                    // them to the real layout rather than replaying
                                    // an intro over a populated map.
                                    if state.nodes.is_empty() {
                                        state.set_nodes(&nodes, sections);
                                    } else {
                                        state.morph_to(&nodes, sections);
                                    }
                                    analyzing = false;
                                    analysis_start = None;
                                    window.request_redraw();
                                }
                                Msg::Error(e) => {
                                    window.set_title(&format!("PulseMap — error: {e}"));
                                    analyzing = false;
                                    elwt.set_control_flow(ControlFlow::Wait);
                                }
                            }
                        }
                    }

                    // Expire timed UI state first, so the checks below see the
                    // truth (these can't live in the else-if chain — a playing
                    // sound would otherwise stop the toast from ever expiring).
                    if let Some((idx, started)) = state.playing {
                        let dur = state.nodes.get(idx).map(|n| n.duration).unwrap_or(0.0);
                        // Held through the release tail so the waveform settles
                        // back to flat instead of snapping.
                        if started.elapsed().as_secs_f32() > dur.max(0.05) + WAVE_RELEASE {
                            state.playing = None;
                        }
                    }
                    if let Some((_, started)) = &state.toast {
                        if started.elapsed() >= TOAST_DURATION {
                            state.toast = None;
                        }
                    }

                    // One buffer rewrite for however many sounds just landed.
                    if instances_dirty {
                        state.write_instances(state.current_intro_t());
                        window.request_redraw();
                    }

                    // Run the force sim; it cools to a stop on its own.
                    let simming = state.tick_physics();
                    if simming {
                        state.write_instances(state.current_intro_t());
                    }

                    // Drive the intro fly-in/colorize animation while it runs.
                    if state.intro_start.is_some() {
                        state.tick_intro();
                        window.request_redraw();
                    } else if simming || state.playing.is_some() || state.toast.is_some() {
                        window.request_redraw();
                    } else if !analyzing {
                        // Settled and idle — stop burning frames.
                        elwt.set_control_flow(ControlFlow::Wait);
                    }
                }

                _ => {}
            }
        })
        .unwrap();
}
