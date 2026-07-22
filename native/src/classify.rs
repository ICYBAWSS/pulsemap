//! Linear classifier head: scale the embedding, one matrix multiply, softmax.
//!
//! Replaces the k-means nearest-prototype classifier, which existed only so the
//! BROWSER could classify without ML libraries. Measured on leakage-safe
//! pack-split CV over 14,526 sounds:
//!
//!     head                      overall   balanced    size
//!     nearest-prototype (old)    76.56%     57.14%    3.9 MB
//!     logistic C=0.1 (this)      79.88%     60.48%    104 KB
//!     RBF SVM C=10               81.08%     58.04%   13.3 MB
//!
//! Balanced accuracy is what matters here — the corpus is ~84:1 imbalanced and
//! the rare classes are the ones users notice getting filed wrong. The linear
//! head wins it outright at a fraction of the size, and its outputs are real
//! probabilities, so the Unsorted cutoff is a probability rather than the
//! hand-rolled distance ratio it replaced.

use std::error::Error;
use std::path::Path;

use serde::Deserialize;

/// Send a sound to Unsorted below this predicted probability.
///
/// Tuned to real feedback, not to a CV optimum: on a 93-sound kit, 19 landed in
/// Unsorted when only 4 belonged there, so the pile must stay small. This puts
/// it at ~5% of a library. Measured over 14.5k sounds, pack-split CV:
///
///     cutoff  rejects  of rejected, % that deserved it  accuracy on filed
///     0.444     3.0%              73.6%                      83.2%
///     0.492     5.0%              68.5%                      84.2%   <- here
///     0.575    10.0%              61.5%                      86.3%
///
/// (the old nearest-prototype head at 5%: 65.5% deserved, 78.8% on filed — this
/// head is better on both at the same pile size.)
const MIN_CONFIDENCE: f32 = 0.49;

#[derive(Deserialize)]
pub struct Model {
    pub labels: Vec<String>,
    scaler_mean: Vec<f32>,
    scaler_scale: Vec<f32>,
    /// [n_classes][embedding_dim]
    coef: Vec<Vec<f32>>,
    intercept: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub section: String,
    pub confidence: f32,
    pub tags: Vec<(String, f32)>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(std::io::BufReader::new(file))?)
    }

    /// `embedding` must already be L2-normalized (as the browser does before
    /// classifying).
    pub fn classify(&self, embedding: &[f32]) -> Classification {
        let scaled: Vec<f32> = embedding
            .iter()
            .zip(&self.scaler_mean)
            .zip(&self.scaler_scale)
            .map(|((&x, &m), &s)| (x - m) / s)
            .collect();

        // logits = coef @ x + intercept
        let logits: Vec<f32> = self
            .coef
            .iter()
            .zip(&self.intercept)
            .map(|(w, &b)| w.iter().zip(&scaled).map(|(a, c)| a * c).sum::<f32>() + b)
            .collect();

        // softmax, shifted by the max for numerical stability
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-9);
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut ranked: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

        let (best, confidence) = ranked[0];
        let section = if confidence >= MIN_CONFIDENCE {
            self.labels[best].clone()
        } else {
            "Unsorted".to_string()
        };
        let tags = ranked
            .iter()
            .take(3)
            .map(|&(i, p)| (self.labels[i].clone(), (p * 1000.0).round() / 1000.0))
            .collect();

        Classification { section, confidence, tags }
    }
}

/// L2-normalize an embedding, matching the browser's `emb / ||emb||` before
/// classification.
pub fn l2_normalize(emb: &[f32]) -> Vec<f32> {
    let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n = if norm > 0.0 { norm } else { 1.0 };
    emb.iter().map(|x| x / n).collect()
}
