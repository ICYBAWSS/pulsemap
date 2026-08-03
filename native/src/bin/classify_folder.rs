// End-to-end check of the SHIPPED pipeline: decode -> trim60 -> mel -> ONNX
// embed -> new 20-class model. Tallies predicted sections for a folder.
use pulsemap::analyzer::{collect_audio, Analyzer};
use std::collections::BTreeMap;
use std::path::Path;
fn main() {
    let assets = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/models"));
    let mut a = Analyzer::new(assets, &assets.join("model.json")).expect("analyzer");
    let root = std::env::args().nth(1).unwrap_or_else(|| "../test_samples".into());
    let files = collect_audio(Path::new(&root));
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    let mut n = 0;
    for f in &files {
        if let Some(r) = a.analyze(f) {
            *tally.entry(r.section).or_default() += 1;
            n += 1;
        }
    }
    println!("classified {n}/{} files:", files.len());
    let mut v: Vec<_> = tally.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for (s, c) in v {
        println!("  {s:<12} {c}");
    }
}
