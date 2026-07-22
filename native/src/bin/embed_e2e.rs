// End-to-end native inference check: waveform → Rust mel preprocessing → ONNX
// (CoreML) → embedding, compared against the PyTorch reference. Proves the full
// chain composes (audio decode is the only remaining upstream piece).
//
// Run from native/:  cargo run --bin embed_e2e

use std::borrow::Cow;
use std::error::Error;

use ndarray::{Array1, Array2, ArrayD};
use ndarray_npy::read_npy;
use ort::execution_providers::CoreML;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use pulsemap::preprocess::input_features;

fn main() -> Result<(), Box<dyn Error>> {
    let wave: Array1<f32> = read_npy("ref/audio_48k_mono.npy")?;
    let mel_filters: Array2<f32> = read_npy("models/mel_slaney.npy")?;
    let feats = input_features(wave.as_slice().unwrap(), &mel_filters); // [1001,64]

    // Shape to the model's [1,1,1001,64].
    let feats = feats.insert_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
    let feats_t = Tensor::from_array(feats.into_dyn())?;

    let mut session = Session::builder()?
        .with_execution_providers([CoreML::default().build()])
        .map_err(|e| e.to_string())?
        .commit_from_file("models/audio_model.onnx")?;

    let in_name = session.inputs()[0].name().to_string();
    let out_name = session.outputs()[0].name().to_string();
    let inputs: Vec<(Cow<str>, SessionInputValue)> =
        vec![(Cow::Owned(in_name), SessionInputValue::from(feats_t))];

    let outputs = session.run(inputs)?;
    let (_shape, data) = outputs[out_name.as_str()].try_extract_tensor::<f32>()?;

    let ref_emb: ArrayD<f32> = read_npy("ref/embedding_ref.npy")?;
    let refv = ref_emb.as_slice().unwrap();

    let n = data.len().min(refv.len());
    let (mut max_abs, mut dot, mut na, mut nb) = (0f32, 0f32, 0f32, 0f32);
    for i in 0..n {
        max_abs = max_abs.max((data[i] - refv[i]).abs());
        dot += data[i] * refv[i];
        na += data[i] * data[i];
        nb += refv[i] * refv[i];
    }
    let cosine = dot / (na.sqrt() * nb.sqrt());
    println!("rust  first5 {:?}", &data[..5]);
    println!("torch first5 {:?}", &refv[..5]);
    println!("\nfull chain vs PyTorch:  max_abs_diff = {max_abs:.3e}   cosine = {cosine:.6}");
    if cosine > 0.9999 && max_abs < 2e-2 {
        println!("PASS \u{2705}  — native waveform→embedding verified");
    } else {
        println!("MISMATCH \u{26a0}\u{fe0f}");
    }
    Ok(())
}
