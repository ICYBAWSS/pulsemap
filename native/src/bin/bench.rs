// Per-stage timing of the analysis pipeline. Run from native/:
//   cargo run --release --bin bench -- "/path/to/folder"
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::Array2;
use pulsemap::analyzer::collect_audio;
use pulsemap::audio::decode_48k_mono;
use pulsemap::classify::{l2_normalize, Model};
use pulsemap::preprocess::input_features;

use std::borrow::Cow;
use ort::ep::coreml::ComputeUnits;
use ort::execution_providers::CoreML;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models");
const PROTOS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../model.json");

fn main() {
    let folder = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../test_samples").to_string()
    }));
    let files = collect_audio(&folder);
    let n = files.len().min(60);
    let files = &files[..n];
    println!("benchmarking {n} files\n");

    let t = Instant::now();
    let mel_filters: Array2<f32> = ndarray_npy::read_npy(Path::new(ASSETS).join("mel_slaney.npy")).unwrap();
    let coreml = CoreML::default()
        .with_compute_units(ComputeUnits::All)
        .with_model_cache_dir(Path::new(ASSETS).join("coreml_cache").to_string_lossy())
        .build();
    let mut session = Session::builder().unwrap()
        .with_execution_providers([coreml])
        .unwrap()
        .commit_from_file(Path::new(ASSETS).join("audio_model.onnx")).unwrap();
    let protos = Model::load(Path::new(PROTOS)).unwrap();
    println!("model load + compile: {:?}\n", t.elapsed());

    let (mut td, mut tp, mut te, mut tc) = (0u128, 0u128, 0u128, 0u128);
    let mut done = 0;
    for f in files {
        let t = Instant::now();
        let Ok(wave) = decode_48k_mono(f) else { continue };
        td += t.elapsed().as_micros();
        let dur = wave.len() as f32 / 48000.0;
        if dur <= 0.0 || dur > 3.0 { continue; }

        let t = Instant::now();
        let feats = input_features(&wave, &mel_filters);
        tp += t.elapsed().as_micros();

        let t = Instant::now();
        let feats = feats.insert_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
        let ft = Tensor::from_array(feats.into_dyn()).unwrap();
        let inn = session.inputs()[0].name().to_string();
        let outn = session.outputs()[0].name().to_string();
        let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![(Cow::Owned(inn), SessionInputValue::from(ft))];
        let out = session.run(inputs).unwrap();
        let (_s, data) = out[outn.as_str()].try_extract_tensor::<f32>().unwrap();
        let emb = data.to_vec();
        te += t.elapsed().as_micros();

        let t = Instant::now();
        let _ = protos.classify(&l2_normalize(&emb));
        tc += t.elapsed().as_micros();
        done += 1;
    }

    let per = |x: u128| x as f64 / done as f64 / 1000.0;
    println!("per-file averages over {done} files (ms):");
    println!("  decode+resample : {:.2}", per(td));
    println!("  mel preprocess  : {:.2}", per(tp));
    println!("  ONNX inference  : {:.2}", per(te));
    println!("  classify        : {:.2}", per(tc));
    println!("  TOTAL           : {:.2}", per(td + tp + te + tc));
}
