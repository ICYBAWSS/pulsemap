//! CLAP audio preprocessing — a faithful port of HF `ClapFeatureExtractor`
//! (unfused: truncation="rand_trunc", padding="repeatpad") + `audio_utils.spectrogram`.
//!
//! Pipeline: 48k mono waveform → repeat-pad to 10s → reflect center-pad →
//! framed periodic-Hann STFT (n_fft 1024, hop 480) → power (|z|²) → slaney mel
//! projection (64 bins) → 10·log10(max(·,1e-10)). Output: [1001, 64] log-mel,
//! exactly the `input_features` the ONNX model expects.
//!
//! Computed in f64 to match numpy (which promotes to float64 for the FFT).

use std::f64::consts::PI;

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

pub const SR: usize = 48_000;
pub const N_FFT: usize = 1024;
pub const HOP: usize = 480;
pub const N_MELS: usize = 64;
pub const MAX_SAMPLES: usize = 480_000; // 10 s
pub const N_FRAMES: usize = 1001;
pub const N_FREQ: usize = N_FFT / 2 + 1; // 513 (one-sided rfft)

/// Periodic Hann window, matching `window_function(1024, "hann", periodic=True)`
/// = `np.hanning(1025)[:-1]` = `0.5 - 0.5·cos(2πn/1024)`.
fn hann_periodic() -> Vec<f64> {
    (0..N_FFT)
        .map(|n| 0.5 - 0.5 * ((2.0 * PI * n as f64) / N_FFT as f64).cos())
        .collect()
}

/// `repeatpad` to MAX_SAMPLES: tile floor(max/len) times, then zero-pad the
/// remainder. Clips longer input (our one-shots are always < 10 s, so the
/// rand-trunc branch never fires; take the head deterministically if it does).
fn repeat_pad(waveform: &[f32]) -> Vec<f64> {
    let len = waveform.len();
    if len == 0 {
        return vec![0.0; MAX_SAMPLES];
    }
    if len >= MAX_SAMPLES {
        return waveform[..MAX_SAMPLES].iter().map(|&x| x as f64).collect();
    }
    let n_repeat = MAX_SAMPLES / len; // floor
    let mut out = Vec::with_capacity(MAX_SAMPLES);
    for _ in 0..n_repeat {
        out.extend(waveform.iter().map(|&x| x as f64));
    }
    out.resize(MAX_SAMPLES, 0.0); // zero-pad to exactly MAX_SAMPLES
    out
}

/// `np.pad(x, pad, mode="reflect")` — mirror around the endpoints without
/// repeating them. Requires `pad < x.len()` (always true here: pad=512).
fn reflect_pad(x: &[f64], pad: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        out.push(x[i]); // left: x[pad], x[pad-1], …, x[1]
    }
    out.extend_from_slice(x);
    for i in 1..=pad {
        out.push(x[n - 1 - i]); // right: x[n-2], x[n-3], …, x[n-1-pad]
    }
    out
}

/// Compute `input_features` [N_FRAMES, N_MELS] from a 48k mono waveform.
/// `mel_filters` is the slaney filterbank, shape [N_FREQ, N_MELS].
pub fn input_features(waveform: &[f32], mel_filters: &Array2<f32>) -> Array2<f32> {
    debug_assert_eq!(mel_filters.shape(), [N_FREQ, N_MELS]);
    let window = hann_periodic();

    // Slaney mel filters are triangular → mostly zero. Precompute each mel's
    // nonzero freq-bin span once so the per-frame projection skips the zeros
    // (~20x fewer mults than the dense 513×64 loop).
    let mut span = [(0usize, 0usize); N_MELS];
    for m in 0..N_MELS {
        let lo = (0..N_FREQ).find(|&k| mel_filters[[k, m]] != 0.0).unwrap_or(0);
        let hi = (0..N_FREQ).rfind(|&k| mel_filters[[k, m]] != 0.0).map(|k| k + 1).unwrap_or(0);
        span[m] = (lo, hi);
    }

    let padded_wave = repeat_pad(waveform);
    let signal = reflect_pad(&padded_wave, N_FFT / 2); // center=True padding

    let num_frames = 1 + (signal.len() - N_FFT) / HOP;
    debug_assert_eq!(num_frames, N_FRAMES);

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buf = vec![Complex::new(0.0f64, 0.0); N_FFT];

    let mut out = Array2::<f32>::zeros((num_frames, N_MELS));
    let mut power = [0.0f64; N_FREQ];

    for frame in 0..num_frames {
        let start = frame * HOP;
        for i in 0..N_FFT {
            buf[i] = Complex::new(signal[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);

        // Power spectrum |z|² over the one-sided bins.
        for k in 0..N_FREQ {
            power[k] = buf[k].norm_sqr();
        }

        // Mel projection + dB: mel[m] = Σ_k fb[k,m]·power[k]; 10·log10(max(·,1e-10)).
        for m in 0..N_MELS {
            let (lo, hi) = span[m];
            let mut acc = 0.0f64;
            for k in lo..hi {
                acc += mel_filters[[k, m]] as f64 * power[k];
            }
            let mel = acc.max(1e-10);
            out[[frame, m]] = (10.0 * mel.log10()) as f32;
        }
    }
    out
}
