// Verify the Rust mel preprocessing reproduces HF's `input_features`.
// Loads the exact waveform Python used (audio_48k_mono.npy) so this isolates
// the preprocessing from any audio-decode differences.
//
// Run from native/:  cargo run --bin mel_check

use std::error::Error;

use ndarray::{Array1, Array2, ArrayD};
use ndarray_npy::read_npy;
use pulsemap::preprocess::{input_features, N_FREQ, N_MELS};

fn main() -> Result<(), Box<dyn Error>> {
    let wave: Array1<f32> = read_npy("ref/audio_48k_mono.npy")?;
    let mel_filters: Array2<f32> = read_npy("models/mel_slaney.npy")?;
    println!("waveform {} samples, filterbank {:?}", wave.len(), mel_filters.shape());
    assert_eq!(mel_filters.shape(), [N_FREQ, N_MELS]);

    let feats = input_features(wave.as_slice().unwrap(), &mel_filters);
    println!("rust input_features {:?}", feats.shape());

    // Reference is [1,1,1001,64]; flatten to compare element-wise.
    let reference: ArrayD<f32> = read_npy("ref/input_features.npy")?;
    let refv = reference.as_slice().unwrap();
    let ours = feats.as_slice().unwrap();
    assert_eq!(refv.len(), ours.len(), "size mismatch");

    let (mut max_abs, mut sum_abs) = (0f32, 0f64);
    let mut argmax = 0usize;
    for i in 0..refv.len() {
        let d = (ours[i] - refv[i]).abs();
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / refv.len() as f64;
    println!("rust  first5 {:?}", &ours[..5]);
    println!("torch first5 {:?}", &refv[..5]);
    println!(
        "at worst idx {argmax}: rust {:.5} vs torch {:.5}",
        ours[argmax], refv[argmax]
    );
    println!("\nmax_abs_diff = {max_abs:.5} dB   mean_abs_diff = {mean_abs:.6} dB");
    if max_abs < 0.5 {
        println!("PASS \u{2705}  — mel preprocessing matches HF");
    } else {
        println!("MISMATCH \u{26a0}\u{fe0f}");
    }
    Ok(())
}
