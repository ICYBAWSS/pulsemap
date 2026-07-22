// Does the 2-D layout actually preserve acoustic similarity?
//
// Metric: k-NN recall — for each sound, what fraction of its k nearest
// neighbours in 512-D CLAP space are still among its k nearest on screen?
// 1.0 = perfect, and a random layout gives ~k/n. This is the objective check
// that the colour gradient means something, since the gradient is read
// straight off these coordinates.
//
//   cargo run --release --bin layout_check
use std::collections::HashSet;
use std::path::PathBuf;

use pulsemap::layout::local_layout_2d_opt;

const K: usize = 10;

fn knn(x: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    (0..x.len())
        .map(|i| {
            let mut d: Vec<(f32, usize)> = (0..x.len())
                .filter(|&j| j != i)
                .map(|j| (x[i].iter().zip(&x[j]).map(|(a, b)| (a - b) * (a - b)).sum(), j))
                .collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0));
            d.truncate(k);
            d.into_iter().map(|(_, j)| j).collect()
        })
        .collect()
}

fn recall(hi: &[Vec<f32>], lo: &[[f32; 2]], k: usize) -> f32 {
    let lo_v: Vec<Vec<f32>> = lo.iter().map(|p| vec![p[0], p[1]]).collect();
    let (a, b) = (knn(hi, k), knn(&lo_v, k));
    let n = hi.len() as f32;
    a.iter()
        .zip(&b)
        .map(|(x, y)| {
            let s: HashSet<_> = x.iter().collect();
            y.iter().filter(|j| s.contains(j)).count() as f32 / k as f32
        })
        .sum::<f32>()
        / n
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Reuse the app's own cache as a real corpus (grouped by assigned section).
    let path = match std::env::var("PM_CACHE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME")?).join(".pulsemap").join("cache.json"),
    };
    let raw = std::fs::read_to_string(&path)?;
    let cache: serde_json::Value = serde_json::from_str(&raw)?;
    let mut by_section: std::collections::BTreeMap<String, Vec<Vec<f32>>> = Default::default();
    for (_, v) in cache.as_object().ok_or("bad cache")? {
        let sec = v["section"].as_str().unwrap_or("?").to_string();
        let emb: Vec<f32> =
            v["embedding"].as_array().ok_or("no emb")?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
        by_section.entry(sec).or_default().push(emb);
    }

    // "packed" = after collision relaxation, i.e. the coordinates actually drawn
    // (and the ones the colour gradient reads). That's the number that matters.
    println!("{:<14} {:>5} {:>9} {:>9} {:>9} {:>8}", "section", "n", "PCA", "neighbor", "packed", "random");
    let (mut tot_pca, mut tot_nn, mut tot_pk, mut tot_n) = (0.0f32, 0.0f32, 0.0f32, 0usize);
    for (sec, embs) in &by_section {
        let n = embs.len();
        if n < 30 {
            continue; // too small for a meaningful k=10 recall
        }
        let pca = local_layout_2d_opt(embs, true);
        let t0 = std::time::Instant::now();
        let nn = local_layout_2d_opt(embs, false);
        let ms = t0.elapsed().as_millis();
        let packed = pulsemap::layout::packed_layout(embs);
        eprintln!("  [{sec}] n={n} neighbour-embedding took {ms} ms");
        let (rp, rn, rk) =
            (recall(embs, &pca, K), recall(embs, &nn, K), recall(embs, &packed, K));
        let rand = K as f32 / n as f32;
        println!(
            "{sec:<14} {n:>5} {:>8.1}% {:>8.1}% {:>8.1}% {:>7.1}%",
            rp * 100.0, rn * 100.0, rk * 100.0, rand * 100.0
        );
        tot_pca += rp * n as f32;
        tot_nn += rn * n as f32;
        tot_pk += rk * n as f32;
        tot_n += n;
    }
    if tot_n > 0 {
        let t = tot_n as f32;
        println!(
            "\nweighted overall:  PCA {:.1}%   neighbor {:.1}%   packed(on-screen) {:.1}%",
            tot_pca / t * 100.0,
            tot_nn / t * 100.0,
            tot_pk / t * 100.0
        );
    }
    Ok(())
}
