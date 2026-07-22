// Milestone 2 verifier: load the CLAP audio ONNX with ORT (CoreML EP), feed the
// Python-produced `input_features`, and check the embedding matches PyTorch.
//
// This decouples the two risks: here we prove ORT + the model + our I/O wiring
// are correct given known-good features. Reproducing `input_features` from raw
// audio in Rust (the mel preprocessing) is the next step, checked separately.
//
// Run from the `native/` dir (after `native/ref/clap_ref.py` has been run):
//   cargo run --bin embed_one

use std::borrow::Cow;
use std::error::Error;

use ndarray::{ArrayD, IxDyn};
use ndarray_npy::read_npy;
use ort::execution_providers::CoreML;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = "models/audio_model.onnx";

    let mut session = Session::builder()?
        .with_execution_providers([CoreML::default().build()])
        .map_err(|e| e.to_string())?
        .commit_from_file(model_path)?;

    println!("== model I/O ==");
    let in_names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
    let out_name = session.outputs()[0].name().to_string();
    for i in session.inputs() {
        println!("  input  {}", i.name());
    }
    for o in session.outputs() {
        println!("  output {}", o.name());
    }

    // Known-good features from Python (shape [1,1,1001,64]).
    let feats: ArrayD<f32> = read_npy("ref/input_features.npy")?;
    println!("input_features shape {:?}", feats.shape());
    let feats_t = Tensor::from_array(feats)?;

    // Our one-shots are <= 3s, so is_longer is always false, shape [1,1].
    let is_longer: ArrayD<bool> = ArrayD::from_elem(IxDyn(&[1, 1]), false);
    let longer_t = Tensor::from_array(is_longer)?;

    // Match tensors to the model's declared input names (order-independent).
    let mut feats_opt = Some(feats_t);
    let mut longer_opt = Some(longer_t);
    let mut inputs: Vec<(Cow<str>, SessionInputValue)> = Vec::new();
    for name in &in_names {
        let val = if name.to_lowercase().contains("longer") {
            SessionInputValue::from(longer_opt.take().expect("one is_longer input"))
        } else {
            SessionInputValue::from(feats_opt.take().expect("one features input"))
        };
        inputs.push((Cow::Owned(name.clone()), val));
    }

    let outputs = session.run(inputs)?;
    let (shape, data) = outputs[out_name.as_str()].try_extract_tensor::<f32>()?;
    println!("\n== embedding ==");
    println!("out '{out_name}' shape {shape:?}");
    println!("rust  first5 {:?}", &data[..5]);

    // Compare against the PyTorch reference.
    let ref_emb: ArrayD<f32> = read_npy("ref/embedding_ref.npy")?;
    let refv = ref_emb.as_slice().expect("contiguous ref");
    println!("torch first5 {:?}", &refv[..5]);

    let n = data.len().min(refv.len());
    let (mut max_abs, mut dot, mut na, mut nb) = (0f32, 0f32, 0f32, 0f32);
    for i in 0..n {
        max_abs = max_abs.max((data[i] - refv[i]).abs());
        dot += data[i] * refv[i];
        na += data[i] * data[i];
        nb += refv[i] * refv[i];
    }
    let cosine = dot / (na.sqrt() * nb.sqrt());
    println!("\nvs PyTorch:  max_abs_diff = {max_abs:.3e}   cosine = {cosine:.6}");
    if cosine > 0.9999 && max_abs < 1e-2 {
        println!("PASS \u{2705}  — ORT + CLAP model + I/O verified");
    } else {
        println!("MISMATCH \u{26a0}\u{fe0f}  — investigate before proceeding");
    }
    Ok(())
}
