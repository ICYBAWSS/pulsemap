//! Analyzer — loads the CLAP model + prototypes + mel filterbank once, then
//! turns an audio file into a classified, embedded `Analyzed` node. Shared by
//! the headless bin and the GUI.

use std::borrow::Cow;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ndarray::Array2;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::audio::decode_48k_mono;
use crate::classify::{l2_normalize, Model};
use crate::layout::Analyzed;
use crate::preprocess::input_features;

const AUDIO_EXTS: &[&str] = &["wav", "aif", "aiff", "flac", "mp3", "ogg", "m4a"];

/// ONNX Runtime's C++ core races on its global type registry
/// (`onnxruntime::data_types_internal::DataTypeRegistry`) when two threads call
/// `commit_from_file` at the same moment — a SIGSEGV in `PlannerImpl`.
///
/// This lock only closes build-vs-build. The registry is ALSO read during
/// inference (`TensorTypeBase::GetElementType`), so building a session while
/// another thread runs `Session::run` crashes the same way — an earlier version
/// of this comment wrongly claimed inference was unaffected, and that cost a
/// second SIGSEGV. `AnalyzerPool::ready` is the barrier that closes
/// build-vs-inference; both are required.
fn session_init_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
const MAX_ONESHOT_SEC: f32 = 3.0;
/// Bars in the playback waveform. Kept small: they're drawn as HUD rects and
/// share the 256-rect instance budget with the legend/panels.
pub const WAVE_BARS: usize = 64;
/// ONNX intra-op threads per session. The worker pool supplies the parallelism,
/// so each session stays narrow — see the measurements in `Analyzer::new`.
pub const INTRA_THREADS: usize = 1;
/// Worker count that paired best with INTRA_THREADS in benchmarking.
pub const BEST_WORKERS: usize = 8;

pub struct Analyzer {
    session: Session,
    protos: Model,
    mel_filters: Array2<f32>,
}

impl Analyzer {
    /// `assets_dir` holds `audio_model.onnx` + `mel_slaney.npy`; `prototypes`
    /// is the path to model.json (the linear classifier head).
    pub fn new(assets_dir: &Path, prototypes: &Path) -> Result<Self, Box<dyn Error>> {
        let mel_filters: Array2<f32> = ndarray_npy::read_npy(assets_dir.join("mel_slaney.npy"))?;
        // CPU execution provider, NOT CoreML. Measured on this model
        // (ms/file, 64-file Jiro pack, M-series):
        // (ms/file, 64-file pack, measured back-to-back on the same machine):
        //   CoreML: 1 worker 77.5 -> 8 workers 61.1
        //   CPU:    1 worker 69.2 -> 8 workers 53.2
        // CPU wins modestly and scales slightly better. Run-to-run spread was
        // 40-45 ms/file on repeats, so treat the gap as "CPU is no worse and
        // probably a bit better", not a precise factor — the test machine had
        // heavy background load throughout.
        //
        // The decisive reason to prefer CPU is fidelity, not speed: it is fp32
        // end-to-end and reproduces the PyTorch reference embedding at cosine
        // 1.00000, which is the space the prototypes were built in. CoreML may
        // compute in fp16 on the ANE.
        //
        // 1 intra-op thread per session: the worker pool supplies parallelism.
        let session = {
            // Poisoning can't happen here (the guarded call can't panic on a
            // valid model file); recover the lock rather than propagate poison.
            let _guard = session_init_lock().lock().unwrap_or_else(|e| e.into_inner());
            Session::builder()?
                .with_intra_threads(INTRA_THREADS)?
                .commit_from_file(assets_dir.join("audio_model.onnx"))?
        };
        let protos = Model::load(prototypes)?;
        Ok(Self { session, protos, mel_filters })
    }

    fn embed(&mut self, feats: Array2<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        let feats = feats.insert_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        let feats_t = Tensor::from_array(feats.into_dyn())?;
        let in_name = self.session.inputs()[0].name().to_string();
        let out_name = self.session.outputs()[0].name().to_string();
        let inputs: Vec<(Cow<str>, SessionInputValue)> =
            vec![(Cow::Owned(in_name), SessionInputValue::from(feats_t))];
        let outputs = self.session.run(inputs)?;
        let (_s, data) = outputs[out_name.as_str()].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    /// Analyze one file; returns None for non-audio, too-long, or undecodable.
    pub fn analyze(&mut self, path: &Path) -> Option<Analyzed> {
        let wave = decode_48k_mono(path).ok()?;
        // One-shot gate uses the RAW duration (matches the browser, which gates
        // before trimming). Trim near-silence to land in the same embedding
        // space the prototypes were trained on (see audio::trim_silence).
        let dur = wave.len() as f32 / 48000.0;
        if dur <= 0.0 || dur > MAX_ONESHOT_SEC {
            return None; // one-shots only, mirroring the browser pipeline
        }
        let trimmed = crate::audio::trim_silence(&wave, 60.0, 1024);
        let feats = input_features(trimmed, &self.mel_filters);
        let raw = self.embed(feats).ok()?;
        let emb = l2_normalize(&raw);
        let cls = self.protos.classify(&emb);
        Some(Analyzed {
            path: path.display().to_string(),
            filename: path.file_name()?.to_string_lossy().into_owned(),
            section: cls.section,
            confidence: cls.confidence,
            embedding: emb,
            envelope: crate::audio::envelope_bars(trimmed, WAVE_BARS),
            duration: trimmed.len() as f32 / 48000.0,
        })
    }
}

/// Recursively collect audio files under a folder (or return the file itself).
pub fn collect_audio(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if is_audio(&p) {
                out.push(p);
            }
        }
    }
    if root.is_dir() {
        walk(root, &mut out);
    } else if is_audio(root) {
        out.push(root.to_path_buf());
    }
    out
}

fn is_audio(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| AUDIO_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}
