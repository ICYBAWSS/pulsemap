// CoreML caps at ~1.7x across threads (measured). Does the plain CPU EP scale
// better? Slower per-file would still win if it scales near-linearly.
use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array2, Array4};
use ndarray_npy::read_npy;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use pulsemap::analyzer::collect_audio;
use pulsemap::audio::{decode_48k_mono, trim_silence};
use pulsemap::preprocess::input_features;

fn build(intra: usize) -> Session {
    Session::builder().unwrap()
        .with_intra_threads(intra).unwrap()
        .commit_from_file("models/audio_model.onnx").unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folder = PathBuf::from(std::env::args().nth(1).expect("pass a folder"));
    let files = collect_audio(&folder);
    let n = files.len().min(32);
    let mel: Array2<f32> = read_npy("models/mel_slaney.npy")?;
    let mut feats = Vec::new();
    for f in &files[..n] {
        let w = decode_48k_mono(f)?;
        feats.push(input_features(trim_silence(&w, 30.0, 1024), &mel));
    }
    let (h, w) = (feats[0].shape()[0], feats[0].shape()[1]);
    println!("{n} files, CPU EP\n");
    println!("{:>8} {:>7} {:>10} {:>10}", "workers", "intra", "ms/file", "speedup");

    let mut base = 0.0f64;
    for (workers, intra) in [(1usize, 8usize), (2, 4), (4, 2), (8, 1)] {
        let mut sessions: Vec<Session> = (0..workers).map(|_| build(intra)).collect();
        let chunk = n.div_ceil(workers);
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for (i, sess) in sessions.iter_mut().enumerate() {
                let part = &feats[(i * chunk).min(n)..((i + 1) * chunk).min(n)];
                s.spawn(move || {
                    let inn = sess.inputs()[0].name().to_string();
                    let outn = sess.outputs()[0].name().to_string();
                    for f in part {
                        let mut arr = Array4::<f32>::zeros((1, 1, h, w));
                        arr.slice_mut(ndarray::s![0, 0, .., ..]).assign(f);
                        let t = Tensor::from_array(arr.into_dyn()).unwrap();
                        let inputs: Vec<(Cow<str>, SessionInputValue)> =
                            vec![(Cow::Owned(inn.clone()), SessionInputValue::from(t))];
                        let out = sess.run(inputs).unwrap();
                        let (_s, d) = out[outn.as_str()].try_extract_tensor::<f32>().unwrap();
                        std::hint::black_box(d.len());
                    }
                });
            }
        });
        let secs = t0.elapsed().as_secs_f64();
        if workers == 1 { base = secs; }
        println!("{workers:>8} {intra:>7} {:>10.1} {:>9.2}x", secs * 1000.0 / n as f64, base / secs);
    }
    Ok(())
}
