// Instanced point-cloud rendering — mirrors the web app's node look:
// soft glow + bright core, confidence -> opacity, hover grows + highlights its
// section with a hue/lightness gradient (relX -> hue, relY -> lightness) while
// dimming everything else.

struct Camera {
    center: vec2<f32>,
    viewport: vec2<f32>,
    zoom: f32,
    point_px: f32,
    hovered: f32,         // instance index of the hovered node, -1 if none
    hovered_section: f32, // section_id (as f32) of the hovered node's section, -1 if none
    time: f32,            // seconds since launch (unused by the point pass now)
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) alpha: f32,
    @location(3) radius_px: f32, // on-screen radius, for constant-width AA
};

fn rgb2hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let l = (mx + mn) * 0.5;
    var h = 0.0;
    var s = 0.0;
    if (mx != mn) {
        let d = mx - mn;
        s = select(d / (2.0 - mx - mn), d / (mx + mn), l <= 0.5);
        if (mx == c.r) { h = (c.g - c.b) / d + select(0.0, 6.0, c.g < c.b); }
        else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
        else { h = (c.r - c.g) / d + 4.0; }
        h /= 6.0;
    }
    return vec3<f32>(h, s, l);
}

fn hue2rgb(p: f32, q: f32, tIn: f32) -> f32 {
    var t = tIn;
    if (t < 0.0) { t += 1.0; }
    if (t > 1.0) { t -= 1.0; }
    if (t < 1.0 / 6.0) { return p + (q - p) * 6.0 * t; }
    if (t < 1.0 / 2.0) { return q; }
    if (t < 2.0 / 3.0) { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    return p;
}

fn hsl2rgb(hsl: vec3<f32>) -> vec3<f32> {
    let h = hsl.x;
    let s = hsl.y;
    let l = hsl.z;
    if (s == 0.0) { return vec3<f32>(l, l, l); }
    let q = select(l + s - l * s, l * (1.0 + s), l < 0.5);
    let p = 2.0 * l - q;
    return vec3<f32>(hue2rgb(p, q, h + 1.0 / 3.0), hue2rgb(p, q, h), hue2rgb(p, q, h - 1.0 / 3.0));
}

@vertex
fn vs_main(
    @builtin(instance_index) inst: u32,
    @location(0) corner: vec2<f32>,     // quad corner in [-1,1]
    @location(1) world: vec2<f32>,      // node position (world)
    @location(2) color: vec3<f32>,      // node base color
    @location(3) confidence: f32,       // 0..1 -> opacity
    @location(4) rel: vec2<f32>,        // relX, relY within its section (0..1)
    @location(5) section_id: u32,
    @location(6) dim: f32,              // search-miss dim multiplier (1.0 = normal, applied to FINAL alpha)
) -> VsOut {
    let is_hover = cam.hovered >= 0.0 && u32(cam.hovered) == inst;
    let has_active_group = cam.hovered_section >= 0.0;
    let in_active_group = has_active_group && f32(section_id) == cam.hovered_section;

    // World-space core radius (grows when you zoom in), pixel-floored so nodes
    // stay visible zoomed out. The quad is 1.8x bigger to give the glow halo
    // room around the core (like the web UI's two-layer node).
    var core_px = max(2.0, cam.point_px * cam.zoom);
    if (is_hover) {
        core_px = core_px * 1.4;
    }
    let size = core_px * 1.8;
    // Positions come from the CPU force sim (layout::physics_step), so they're
    // already the settled/settling coordinates — no shader-side drift, which
    // would fight the sim and desync hit-testing.
    let center_px = (world - cam.center) * cam.zoom;
    let px = center_px + corner * size;
    let ndc = px / (cam.viewport * 0.5);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.corner = corner;

    var col = color;
    if (in_active_group) {
        let hsl = rgb2hsl(color);
        let shiftedHue = fract(hsl.x + (rel.x - 0.5) * (60.0 / 360.0) + 1.0);
        let shiftedL = clamp(hsl.z + (rel.y - 0.5) * 0.30, 0.25, 0.85);
        col = hsl2rgb(vec3<f32>(shiftedHue, hsl.y, shiftedL));
    }
    out.color = col;

    var a = 0.5 + 0.5 * confidence;
    if (has_active_group && !in_active_group) {
        a = a * 0.15;
    }
    a = a * dim; // search-miss dimming — multiplies the FINAL alpha, not confidence
    // (folding it into `confidence` pre-floor was the bug: the 0.5 floor above
    // swallowed most of the dim, so search barely did anything visually)
    if (is_hover) {
        a = 1.0;
    }
    out.alpha = a;
    out.radius_px = size;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.corner);
    if (d > 1.0) {
        discard;
    }
    // Glowing orb: a crisp bright core (inner 1/1.8 of the quad) plus a soft
    // halo that falls off to the edge. Core edge AA is a constant ~1.5px so it
    // stays sharp at any zoom.
    let core_frac = 1.0 / 1.8;
    let aa = clamp(1.5 / in.radius_px, 0.001, 0.4);
    let core = 1.0 - smoothstep(core_frac - aa, core_frac + aa, d);
    let halo = pow(clamp(1.0 - d, 0.0, 1.0), 1.6) * 0.55;
    let intensity = clamp(core + halo, 0.0, 1.0);
    if (intensity <= 0.0) {
        discard;
    }
    // Lift the core slightly toward white so it reads as a lit orb.
    let lit = mix(in.color, vec3<f32>(1.0), core * 0.18);
    return vec4<f32>(lit, intensity * in.alpha);
}
