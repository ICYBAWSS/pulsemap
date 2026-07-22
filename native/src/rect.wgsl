// Screen-space rounded-rect instances: panel backgrounds, progress bars,
// legend swatches/hover-highlight. Independent of the world camera — pos/size
// are in physical pixels, origin top-left (matches winit cursor coords).

struct Screen {
    size: vec2<f32>, // viewport, physical px
};
@group(0) @binding(0) var<uniform> screen: Screen;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,   // pixel offset from rect center
    @location(1) half_size: vec2<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) corner: vec2<f32>,   // [-1,1] quad corner
    @location(1) pos: vec2<f32>,      // top-left, px
    @location(2) size: vec2<f32>,     // width/height, px
    @location(3) radius: f32,         // corner radius, px
    @location(4) color: vec4<f32>,
) -> VsOut {
    let half_size = size * 0.5;
    let center = pos + half_size;
    let px = center + corner * half_size;
    // NDC: x right+, y up+; screen px is y-down from top-left.
    let ndc = vec2<f32>(
        (px.x / screen.size.x) * 2.0 - 1.0,
        1.0 - (px.y / screen.size.y) * 2.0,
    );
    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.local = corner * half_size;
    out.half_size = half_size;
    out.radius = radius;
    out.color = color;
    return out;
}

// Signed distance to a rounded box, for a crisp anti-aliased edge.
fn sd_round_box(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = sd_round_box(in.local, in.half_size, in.radius);
    let aa = 1.0;
    let alpha = 1.0 - smoothstep(-aa, aa, d);
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
