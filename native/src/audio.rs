//! Audio decode → 48k mono f32, via symphonia (any format) + a windowed-sinc
//! resampler. Mirrors `librosa.load(path, sr=48000, mono=True)`: average
//! channels to mono, then resample to 48 kHz.
//!
//! The resampler isn't bit-identical to librosa's soxr (that's fine — the CLAP
//! embedding is robust to sub-perceptual resampling differences; we verify the
//! resulting embedding's cosine against the reference), but it's a proper
//! anti-aliased windowed-sinc, not linear interpolation.

use std::error::Error;
use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::preprocess::SR;

/// Trim leading/trailing near-silence — an exact port of the web app's
/// `trimSilence` / `embed.py`'s `js_trim_silence` (top_db=30, frame=1024, RMS
/// per frame vs a peak-relative threshold). The prototypes were trained on
/// TRIMMED audio, so skipping this puts the query embedding in a different
/// space (median cosine ~0.76 vs the trained space) and wrecks classification.
pub fn trim_silence(samples: &[f32], top_db: f32, frame: usize) -> &[f32] {
    let n = samples.len();
    if n == 0 {
        return samples;
    }
    let peak = samples.iter().fold(0f32, |m, &x| m.max(x.abs()));
    if peak == 0.0 {
        return samples;
    }
    let threshold = peak * 10f32.powf(-top_db / 20.0);
    let rms = |s: usize, e: usize| -> f32 {
        let cnt = (e - s).max(1);
        (samples[s..e].iter().map(|&x| x * x).sum::<f32>() / cnt as f32).sqrt()
    };

    let mut start = 0;
    let mut i = 0;
    while i < n {
        if rms(i, (i + frame).min(n)) > threshold {
            start = i;
            break;
        }
        i += frame;
    }
    let mut end = n;
    let mut i = n;
    while i > 0 {
        let s = i.saturating_sub(frame);
        if rms(s, i) > threshold {
            end = i;
            break;
        }
        i = i.saturating_sub(frame);
    }
    if start < end {
        &samples[start..end]
    } else {
        samples
    }
}

/// Peak-amplitude envelope in `bars` buckets, normalized to 0..1. This is what
/// the playback waveform draws, so the animation reflects THIS sound's actual
/// shape (a snare's sharp spike vs an open hat's long decay) rather than a
/// generic pulse. Computed once at analyze time off the already-decoded wave.
pub fn envelope_bars(samples: &[f32], bars: usize) -> Vec<f32> {
    if samples.is_empty() || bars == 0 {
        return Vec::new();
    }
    let per = (samples.len() as f32 / bars as f32).max(1.0);
    let mut out: Vec<f32> = (0..bars)
        .map(|i| {
            let s = (i as f32 * per) as usize;
            let e = (((i + 1) as f32 * per) as usize).min(samples.len());
            samples[s..e.max(s + 1).min(samples.len())]
                .iter()
                .fold(0f32, |m, &x| m.max(x.abs()))
        })
        .collect();
    let peak = out.iter().fold(0f32, |m, &x| m.max(x)).max(1e-6);
    for v in &mut out {
        *v /= peak;
    }
    out
}

/// Decode any supported audio file to a 48 kHz mono f32 waveform.
pub fn decode_48k_mono(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or("no default audio track")?
        .clone();
    let track_id = track.id;
    let in_sr = track.codec_params.sample_rate.ok_or("unknown sample rate")?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1)
        .max(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    // Accumulate a mono sum (average of channels).
    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                if sample_buf.is_none() {
                    sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
                }
                let sbuf = sample_buf.as_mut().unwrap();
                sbuf.copy_interleaved_ref(decoded);
                let samples = sbuf.samples();
                // Interleaved [ch0,ch1,...] per frame → average to mono.
                for frame in samples.chunks(channels) {
                    let sum: f32 = frame.iter().sum();
                    mono.push(sum / channels as f32);
                }
            }
            Err(SymError::DecodeError(_)) => continue, // skip a bad packet
            Err(e) => return Err(e.into()),
        }
    }

    Ok(if in_sr == SR as u32 {
        mono
    } else {
        resample(&mono, in_sr, SR as u32)
    })
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Windowed-sinc (Blackman) resampler, arbitrary ratio. On downsampling the
/// kernel cutoff drops to the output Nyquist so we anti-alias.
///
/// The kernel is precomputed into a polyphase table (PHASES sub-sample offsets ×
/// TAPS weights) so the hot loop is pure multiply-add — no sin/cos per sample,
/// which was ~58% of analysis time.
fn resample(x: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    if in_sr == out_sr || x.is_empty() {
        return x.to_vec();
    }
    let ratio = out_sr as f64 / in_sr as f64; // output-per-input
    let out_len = ((x.len() as f64) * ratio).round() as usize;
    let cutoff = if ratio < 1.0 { ratio } else { 1.0 };
    const HALF: i64 = 32; // kernel half-width in input samples
    const TAPS: usize = (HALF * 2) as usize;
    const PHASES: usize = 512;

    // Build the table once: row p is fractional offset frac = p/PHASES; tap i
    // maps to m = i-(HALF-1), tau = frac-m. Matches the old per-sample math.
    let mut table = vec![0f64; PHASES * TAPS];
    for p in 0..PHASES {
        let frac = p as f64 / PHASES as f64;
        for i in 0..TAPS {
            let m = i as i64 - (HALF - 1);
            let tau = frac - m as f64;
            let wn = std::f64::consts::PI * tau / HALF as f64;
            let window = 0.42 + 0.5 * wn.cos() + 0.08 * (2.0 * wn).cos();
            table[p * TAPS + i] = cutoff * sinc(cutoff * tau) * window;
        }
    }

    let mut out = vec![0f32; out_len];
    for (j, o) in out.iter_mut().enumerate() {
        let t = j as f64 / ratio;
        let center = t.floor() as i64;
        let frac = t - center as f64;
        let p = ((frac * PHASES as f64) as usize).min(PHASES - 1);
        let row = &table[p * TAPS..(p + 1) * TAPS];
        let (mut acc, mut wsum) = (0f64, 0f64);
        for (i, &w) in row.iter().enumerate() {
            let k = center + i as i64 - (HALF - 1);
            if k < 0 || k as usize >= x.len() {
                continue;
            }
            acc += x[k as usize] as f64 * w;
            wsum += w;
        }
        *o = if wsum.abs() > 1e-12 { (acc / wsum) as f32 } else { 0.0 };
    }
    out
}
