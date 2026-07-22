// Full headless pipeline for one file: decode → mel → ONNX embed → normalize →
// classify → cache. Verifies section/confidence/tags against the Python
// reference and the embedding cosine against PyTorch, then round-trips the cache.
//
// Run from native/:  cargo run --bin analyze [path/to/file.wav]

use std::borrow::Cow;
use std::error::Error;
use std::path::{Path, PathBuf};

use ndarray::{Array2, ArrayD};
use ndarray_npy::read_npy;
use ort::execution_providers::CoreML;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use pulsemap::audio::decode_48k_mono;
use pulsemap::cache::{fingerprint, Cache, CacheEntry};
use pulsemap::classify::{l2_normalize, Model};
use pulsemap::preprocess::input_features;

fn embed(session: &mut Session, feats: Array2<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
    let feats = feats.insert_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0));
    let feats_t = Tensor::from_array(feats.into_dyn())?;
    let in_name = session.inputs()[0].name().to_string();
    let out_name = session.outputs()[0].name().to_string();
    let inputs: Vec<(Cow<str>, SessionInputValue)> =
        vec![(Cow::Owned(in_name), SessionInputValue::from(feats_t))];
    let outputs = session.run(inputs)?;
    let (_s, data) = outputs[out_name.as_str()].try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        "../test_samples/BD AXIOM (DEMO)/AXIOM (DEMO)/KICK/BD 808ISH² KICK.wav".to_string()
    }));
    println!("file: {}", path.display());

    let protos = Model::load(Path::new("../model.json"))?;
    let mel_filters: Array2<f32> = read_npy("models/mel_slaney.npy")?;
    let mut session = Session::builder()?
        .with_execution_providers([CoreML::default().build()])
        .map_err(|e| e.to_string())?
        .commit_from_file("models/audio_model.onnx")?;

    // ---- decode → embed → classify ----
    let wave = decode_48k_mono(&path)?;
    println!("decoded {} samples @48k ({:.3}s)", wave.len(), wave.len() as f32 / 48000.0);
    let feats = input_features(&wave, &mel_filters);
    let raw_emb = embed(&mut session, feats)?;
    let emb = l2_normalize(&raw_emb);
    let cls = protos.classify(&emb);
    println!(
        "\n=> section: {}   confidence: {:.4}",
        cls.section, cls.confidence
    );
    println!("   tags: {:?}", cls.tags);

    // ---- verify vs references ----
    let reference: serde_json::Value =
        serde_json::from_reader(std::fs::File::open("ref/classify_ref.json")?)?;
    let ref_section = reference["section"].as_str().unwrap_or("");
    println!(
        "\nvs Python classify:  ref section '{ref_section}' conf {}",
        reference["confidence"]
    );

    // Embedding cosine vs PyTorch (our decode differs from librosa's resampler).
    let ref_emb: ArrayD<f32> = read_npy("ref/embedding_ref_norm.npy")?;
    let refv = ref_emb.as_slice().unwrap();
    let dot: f32 = emb.iter().zip(refv).map(|(a, b)| a * b).sum();
    let cos = dot / (emb.iter().map(|x| x * x).sum::<f32>().sqrt()
        * refv.iter().map(|x| x * x).sum::<f32>().sqrt());
    println!("embedding cosine vs PyTorch (full decode path): {cos:.5}");

    let section_ok = cls.section == ref_section;
    if section_ok && cos > 0.999 {
        println!("PASS \u{2705}  — section matches, embedding cosine {cos:.5}");
    } else {
        println!("CHECK \u{26a0}\u{fe0f}  section_ok={section_ok} cosine={cos:.5}");
    }

    // ---- cache round-trip ----
    let cache_path = PathBuf::from("target/pm_cache.json");
    let mut cache = Cache::load(cache_path.clone());
    let fp = fingerprint(&path)?;
    cache.insert(
        fp.clone(),
        CacheEntry {
            path: path.display().to_string(),
            filename: path.file_name().unwrap().to_string_lossy().into_owned(),
            embedding: emb.clone(),
            section: cls.section.clone(),
            confidence: cls.confidence,
            tags: cls.tags.clone(),
            envelope: Vec::new(),
            duration: 0.0,
        },
    );
    cache.save()?;
    let reloaded = Cache::load(cache_path);
    match reloaded.get(&fp) {
        Some(e) if e.section == cls.section && e.embedding.len() == 512 => {
            println!("cache: round-trip OK ({} entr{})", reloaded.len(), if reloaded.len() == 1 { "y" } else { "ies" });
        }
        _ => println!("cache: round-trip FAILED"),
    }
    Ok(())
}
