# PulseMap — Native Desktop App Plan

Planning notes for rewriting PulseMap as a native, cross-platform desktop
application. Supersedes the browser/`file://` and webview approaches.

**Decided:** Rust, end-to-end, **cross-platform (macOS + Windows, Linux nice-to-have).**
Chosen on objective engineering merit — not reuse, familiarity, or ease.

---

## Why Rust (and not the alternatives)

The app is four demanding native subsystems: **neural inference**, **large-scale
interactive point-cloud rendering + force simulation**, **real-time audio
audition**, and the **DSP/UMAP glue**. Rust is the one language where all four are
first-class and native in a single type-safe, GPU-capable, truly-parallel binary —
no IPC seams, no runtime tax, no GC pauses, no webview.

- **Not PyTorch/Python** — great for research, wrong for *deploying* a fixed model
  (heavy runtime, GIL, slow startup, huge bundle).
- **Not Electron/Tauri-webview** — trades peak rendering performance for UI reuse,
  which we're explicitly ignoring.
- **Not C++** — same ceiling as Rust but memory-unsafe, slower iteration, zero
  upside on a greenfield project.
- **Caveat (not our case):** if this were **macOS-only**, Swift + MLX/CoreML +
  Metal would win on peak Apple-silicon performance. Cross-platform rules it out.

---

## Target stack

| Layer | Choice | Notes |
|---|---|---|
| **Inference** (dominant cost) | **ONNX Runtime via `ort`** | Execution providers per platform: **CoreML** (macOS), **DirectML** (Windows), **CUDA/TensorRT** (Linux/NVIDIA), CPU fallback everywhere. Loads CLAP ONNX at **fp32 = PyTorch-equivalent accuracy**. |
| **Rendering + force sim** | **`wgpu`** | GPU-instanced point quads; **compute-shader** N-body / Barnes-Hut force layout. Target 100k+ nodes @ 60fps. `egui` (or custom wgpu UI) for panels/controls. |
| **Audio audition** | **`symphonia`** (decode) + **`cpal`** (playback) | wav/aif/flac/mp3/ogg/m4a. Sub-ms hover-to-audition trigger. |
| **DSP features** | `rustfft` + hand-rolled | Port the biquad/zero-crossing feature extractor. |
| **Similarity layout** | `hnsw` (approx. kNN) + a neighbor-preserving embedder | **Algorithm is not fixed to UMAP** — see below. `hnsw` gives the kNN graph either way. |
| **Parallelism** | `rayon` | Batch-embed the whole library across all cores. |
| **Packaging** | `cargo` + per-OS bundlers | Single self-contained binary; codesign/notarize (mac), MSI/exe (win). |

---

## What the app must achieve (feature parity + goals)

### Core pipeline (port from `build_map.py` / `pipeline.js`)
1. **Ingest** a dropped/selected folder; recurse; keep audio by extension.
2. **Decode → mono → 48kHz**, trim leading/trailing silence.
3. **One-shots only** — keep clips ≤ 3.0s (loops are out of scope for now).
4. **Embed** each clip with CLAP (`laion/clap-htsat-unfused`), L2-normalized
   `get_audio_features` pooler output.
5. **Classify** by nearest prototype (from `prototypes.json`): assign section,
   margin → confidence, below-threshold → **Unsorted**.
6. **Layout**:
   - **Requirement (not the algorithm):** within each group, position nodes so
     **similar sounds sit next to similar sounds** — a similarity-preserving 2D
     embedding. UMAP is *a* candidate, not a mandate. Options to evaluate
     empirically: **UMAP, t-SNE, PaCMAP, MDS, spectral layout, or a
     force-directed layout over the kNN similarity graph.** Pick whichever gives
     the cleanest "neighbors sound alike" result at our group sizes; the
     invariant is the outcome, not the method.
   - Compute it over the **CLAP embeddings** (NOT the 13 DSP features — that was
     the regression that made neighbors not sound alike). For UMAP specifically,
     `n_neighbors = min(15, n-1)`, `min_dist ≈ 0.12`. Small-n fallback (e.g. a
     circle) for groups too small to embed meaningfully.
   - Sections arranged on a grid, **ordered by 1D PCA of section centroids** so
     acoustically-adjacent families sit next to each other.
7. **Cache** embeddings by file fingerprint (name|size|mtime) for instant reloads.

### UX / feature parity — HARD REQUIREMENT
**The native app must reproduce the current browser app's UX and full feature
set — same interactions, same feel.** This is a port, not a redesign. The
rendering tech underneath changes (wgpu instead of canvas/d3); the experience
does not. Complete checklist, ported from `app.js` / `pipeline.js` / `index.html`,
with the native mechanism for each:

**Entry & loading**
- [ ] Drop-zone front door: drag a folder in **or** pick one; recurse the tree,
      keep audio by extension. → native folder picker + drag-drop + `walkdir`.
- [ ] Progress UI while analyzing: label + progress bar + running count. → egui.
- [ ] Loading overlay → drop-zone hidden once the map renders.

**Map rendering & motion**
- [ ] Section-grouped layout; similarity layout within each group.
- [ ] **Intro reveal:** nodes start grey, fly to home, colorize on arrival
      (one-way grey → section color). → per-node tween in the vertex shader.
- [ ] Force sim: spring-to-home + collision + gentle repulsion, "breathing"
      clusters; params scale with node count (spring/decay/cooldown/collide).
      → GPU compute-shader force sim (Milestone 5).
- [ ] Node look: soft glow + core, **confidence → opacity**, hover grows the
      node. → extend `point.wgsl`.
- [ ] **Section color** by family; **within-section gradient** (layout-x → hue,
      layout-y → lightness) shown for the active/hovered group.
- [ ] Auto-fit framing (zoom/center to fit all on load + resize).

**Interaction**
- [ ] **Hover** → audition the sound (native playback) + banner
      "filename · section · confidence%"; node grows; stop on leave.
- [ ] Zoom / pan (done in M1).
- [ ] **Drag a node into another section** → reclassify: recolor, set new home,
      spring settles; invalid drop springs back. Records a correction, toast
      "✓ moved to X", updates counts + export.
- [ ] Drag-target ring indicator near a section while dragging.
- [ ] **Legend / navigator** sidebar: swatch + name + count per section; click a
      row → fly-to (center + zoom); hover a row → highlight that group, dim
      others. → egui side panel.
- [ ] **Search** box: filter by filename / section / tags; misses dimmed.
- [ ] Group dimming: when a group is active/hovered, other groups dim.

**Data & persistence**
- [ ] Per-file **embedding cache** keyed by fingerprint (name|size|mtime). →
      on-disk cache in the app-data dir (replaces IndexedDB).
- [ ] Last map layout cached for instant reopen (replaces localStorage).
- [ ] **Corrections** persisted + **Export** as `corrections.json` for retraining.
- [ ] Toast notifications; window resize handling.
- [ ] **Privacy:** audio never leaves the machine (already true; stays true).

### Non-functional goals
- **Cross-platform:** macOS + Windows first-class; Linux best-effort.
- **GPU-accelerated** inference and rendering on each platform.
- **Fast:** GPU inference >> current WASM/q8; smooth at large library sizes.
- **Accurate:** fp32 model (better clustering than the browser's q8).
- **Self-contained:** one installable binary, no Python/Node/Deno, no terminal,
  no browser, no `file://` CSP issues (the whole reason we're leaving the web).
- **Local-first / private:** audio never leaves the machine.

### Model distribution — DECIDED: download-on-first-run
- Ship a **small binary**; fetch the CLAP ONNX weights on first launch, cache to
  a per-user app-data dir, verify with a **checksum**, and reuse thereafter.
- Needs: a first-run progress UI, a pinned download URL + hash, graceful
  offline/failure handling and retry. (Only first run needs network; the model
  is the only remote asset — audio always stays local.)

---

## Assets we can carry over (data, not code)
- `prototypes.json` — the classifier prototypes (per-section embedding centroids
  + tags). Reusable as-is; just load into Rust.
- CLAP model — export/confirm an **ONNX** graph for `ort` (Xenova already has an
  ONNX export; verify fp32 vs the q8 we were using in-browser).
- The color palette (section → family color) and UMAP params from `build_map.py`.
- The 13-dim acoustic-feature definition (if we keep it for anything).

## Open questions / to decide later
- UI framework detail: `egui`-on-`wgpu` for panels vs a fully custom wgpu scene.
- Similarity-layout algorithm: which of UMAP / t-SNE / PaCMAP / MDS / spectral /
  force-on-kNN wins empirically at our group sizes — and crate vs implement.
- Do we keep acoustic features at all, or go pure-CLAP for both layout and color?
- Windows GPU: DirectML EP availability/packaging specifics.

## Rough milestones
1. ✅ **Skeleton:** `wgpu` window rendering a static point cloud + pan/zoom-to-cursor.
   Done — `native/` (Rust 1.97, wgpu 0.19 + winit 0.29, Metal backend verified).
2. ✅ **Inference:** `ort` (2.0-rc.12, CoreML EP) loads `audio_model.onnx` and
   embeds via `input_features` → `audio_embeds` [1,512]. Verified against PyTorch
   (`native/src/bin/embed_one.rs` vs `native/ref/clap_ref.py`): **cosine 1.000000**,
   max abs diff 9.7e-5. Note: unfused export has a single input (no `is_longer`).
3. **Pipeline:** folder → decode → embed → classify → cache (headless, verified
   against `build_map.py` outputs on the same folder).
   - ✅ **Preprocessing + embed:** `native/src/preprocess.rs` ports HF
     `ClapFeatureExtractor` (repeatpad → reflect center-pad → periodic-Hann STFT
     n_fft 1024/hop 480 → power → slaney mel 64 → 10·log10). Mel matches HF to
     1e-5 dB (`mel_check`); full waveform→embedding matches PyTorch cosine
     1.000000 (`embed_e2e`). Slaney filterbank baked as `models/mel_slaney.npy`.
   - ✅ **Audio decode:** `native/src/audio.rs` — symphonia (all formats) → mono
     → windowed-sinc resample to 48k. Real 44.1k wav → embedding cosine 1.00000.
   - ✅ **Classify:** `native/src/classify.rs` ports `pipeline.js` nearest-prototype
     (scale → per-label min Euclidean → margin→confidence, 0.20 → Unsorted).
     Matches Python reference exactly; generalizes across sections.
   - ✅ **Cache:** `native/src/cache.rs` — on-disk JSON keyed by name|size|mtime.
   - **M3 fully verified via `native/src/bin/analyze.rs` (file → section).**
4. **Layout:** UMAP per section + centroid-PCA grid arrangement.
4b. ✅ **Native workflow wired:** drop folder → background analysis thread →
   section layout (PCA first-pass) → GPU point cloud, camera auto-fit. Hover →
   **audio audition** (rodio) + node grow + banner (in title bar for now);
   confidence → opacity. `native/src/{layout,analyzer}.rs`, `main.rs`.
5. **Viz / full parity (in progress):** DONE: folder-drop, progress, section
   layout+color (organic sunflower-packed blobs, NOT similarity-ordered per
   user), auto-fit, hover-audition, confidence→opacity, hover-grow, pan/zoom,
   glow (bright-core+halo, constant-px AA), category-region hover (gradient +
   grey-others), planet/galaxy collision-relaxed layout, model preload pool
   (loads at launch, not per-drop), progress ETA.
   - ✅ **wgpu 0.19 → 30.0.0 migration** (2026-07-15), done to unlock on-canvas
     text (glyphon needs wgpu ≥22; no version pairs with 0.20). Verified via
     rustc's exhaustive error list against real wgpu 30 source (not guessed):
     `CurrentSurfaceTexture` enum replaces `Result<_, SurfaceError>` from
     `get_current_texture()`, `Queue::present()` replaces `SurfaceTexture::present()`,
     `entry_point`/vertex `buffers` now `Option`-wrapped, new required fields
     (`apply_limit_buckets`, `color_space`, `depth_slice`, `multiview_mask`,
     `cache`, `immediate_size` replacing `push_constant_ranges`). Confirmed
     single wgpu version tree-wide (wgpu/core/hal/types all 30.0.0, glyphon
     0.12.0) — no duplicate/conflicting copies. Runtime-verified, not just
     compiled. `native/Cargo.toml`, `native/src/main.rs`.
   - ✅ **Text HUD wired** (2026-07-15): `native/src/hud.rs` (`Hud` struct) +
     `native/src/rect.wgsl` (screen-space rounded-rect instances for panels/
     bars/swatches, SDF edge AA). Real font shaping/kerning via glyphon+
     cosmic-text. Shipped: title, progress panel (live bar + count + ETA),
     top-center hover banner (idle text when nothing hovered), top-right
     section legend (swatch+name+count, hover-highlights the section on the
     map via the same gradient/dim path canvas-hover uses, click-to-fly
     snaps camera to the section). `Msg::Done` now carries `SectionMeta`
     through to `State` for the legend. Runtime-verified (launches, no
     errors); palette matches the old web app's CSS vars.
   - ✅ **Search box** (2026-07-21): top-of-window text field, always
     "focused" (only text field in the app — no explicit click-to-focus).
     Real keyboard text input via `WindowEvent::KeyboardInput`'s
     `KeyEvent.text` (layout-aware typed chars), Backspace to delete, Escape
     clears the query first / quits only when already empty. Filters by
     filename/section substring (case-insensitive), non-matches dimmed to
     12% opacity — mirrors the web app's search-miss dimming. Implemented by
     baking the dim factor into each node's GPU-instance `confidence` at
     write time (`State::search_dim`), so no shader change was needed;
     re-applied on every `write_instances` call including mid-intro-animation
     (via `current_intro_t()`, so typing never skips the reveal animation).
   - ✅ **Drag-to-reclassify** (2026-07-21): press-drag a node onto another
     section to reclassify it — mirrors the web app's core editing feature.
     Press distinguishes legend-click / node-grab / canvas-pan by hit-testing
     in that order. While dragging, the grabbed node follows the cursor and a
     ring indicator (`world_to_screen` + two nested rounded rects) shows the
     nearest section within `DROP_RADIUS` (20 world units). On release: valid
     drop recolors + reclassifies the node, updates both sections' counts,
     shows a 2.5s auto-expiring toast ("✓ moved to X"), and records a
     correction; invalid drop springs back to `drag_origin`. One deliberate
     simplification vs. the web app: without a continuous force sim, a
     dropped node just stays exactly where dropped (may overlap neighbors)
     rather than being nudged apart by ongoing physics — matches the web
     app's stated behavior ("its new anchor = where you dropped it"), just
     without the subsequent organic settling.
   - ✅ **Corrections persisted automatically** (no manual "Export" button):
     every reclassification writes `~/.pulsemap/corrections.json` immediately
     — the native equivalent of the web app's manual export action, chosen
     over building button-hit-test UI for a single one-off action.
   - ✅ **On-disk embedding cache wired into the GUI** (2026-07-21): the
     `cache.rs` `Cache`/`fingerprint` API existed but wasn't actually called
     anywhere — every drop re-ran the full model on every file, even on a
     repeat drop of the same unchanged folder. Now `AnalyzerPool` loads
     `~/.pulsemap/cache.json` once at startup (shared `Arc<Mutex<Cache>>`
     across workers), each worker checks it before running the model and
     inserts on a miss, and the batch saves once at the end (not per-file, to
     avoid I/O thrashing). Re-dropping an unchanged folder now skips
     decode+embed+classify entirely for every cached file — the native
     equivalent of the web app's IndexedDB cache / instant-reload behavior.
   - ⬜ Still TODO: compute-shader force sim (continuous "breathing"
     collision physics — the layout is currently static collision-relaxed at
     build time, not continuously simulated; this is the single largest
     remaining lift, deliberately deferred as its own chunk), animated
     (not instant-snap) legend-click camera fly-to. NOTE: accuracy/model
     work paused per user — parity first.
   - ✅ **Fixed a real crash** (2026-07-21): SIGSEGV in ONNX Runtime's C++ core
     (`onnxruntime::data_types_internal::DataTypeRegistry`) — `AnalyzerPool::new`
     spawns N worker threads that each call `Session::builder()...commit_from_file()`
     at startup; ORT's session-planning code isn't thread-safe for *concurrent
     construction* across sessions (a known class of ORT issue), so two threads
     initializing at once raced on ORT's internal global type registry. Root
     cause confirmed via the crash log's own frame naming the global singleton.
     Fix: `native/src/analyzer.rs` serializes `commit_from_file` behind a
     process-wide `OnceLock<Mutex<()>>`; inference (`Session::run`) on already-
     built sessions is unaffected and still fully parallel across workers.
     5/5 stress-launches clean after the fix (race was intermittent before, so
     this is strong evidence, not absolute proof — flag if it recurs).
   - ✅ **Startup screen** + **background color fix**: centered "PulseMap" +
     instructions when idle/empty (mirrors the web dropzone copy); clear color
     changed from blue-tinted navy to neutral charcoal (user disliked the tint).
   - ✅ **DPI/HiDPI fix**: all HUD sizes/positions were specified as if physical
     px == logical px, so text rendered at half size on Retina. Added
     `State.ui_scale` (from `window.scale_factor()`, updated on
     `ScaleFactorChanged`), threaded through every HUD literal *and* through
     `hud::legend_layout`'s hit-testing (so clicks stay aligned with what's
     drawn) — a bug that only bites at scale≠1.0 and is easy to reintroduce if
     new HUD elements skip the `* s` multiplier.
6. **Polish + package:** codesign/notarize, installers, model download-on-first-run.
