// Does the worker pool actually give parallel speedup, or does CoreML serialize
// inference across sessions? Runs the same file set at 1, 2, 4, 8 workers.
//   cargo run --release --bin bench_par -- "/path/to/folder"
use std::path::{Path, PathBuf};
use std::time::Instant;

use pulsemap::analyzer::{collect_audio, Analyzer};

const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models");
const PROTOS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../model.json");

fn main() {
    let folder = PathBuf::from(std::env::args().nth(1).expect("pass a folder"));
    let all = collect_audio(&folder);
    let n = all.len().min(64);
    let files = &all[..n];
    println!("{n} files per run\n");
    println!("{:>7} {:>10} {:>12} {:>9}", "workers", "wall (s)", "ms/file", "speedup");

    let mut base = 0.0f64;
    for w in [1usize, 2, 4, 8] {
        // Build sessions first so model load isn't timed.
        let mut analyzers: Vec<Analyzer> = (0..w)
            .map(|_| Analyzer::new(Path::new(ASSETS), Path::new(PROTOS)).expect("analyzer"))
            .collect();

        let chunk = n.div_ceil(w);
        let t = Instant::now();
        std::thread::scope(|s| {
            for (i, an) in analyzers.iter_mut().enumerate() {
                let part = &files[(i * chunk).min(n)..((i + 1) * chunk).min(n)];
                s.spawn(move || {
                    for f in part {
                        let _ = an.analyze(f);
                    }
                });
            }
        });
        let secs = t.elapsed().as_secs_f64();
        if w == 1 {
            base = secs;
        }
        println!(
            "{w:>7} {:>10.2} {:>12.1} {:>8.2}x",
            secs,
            secs * 1000.0 / n as f64,
            base / secs
        );
    }
}
