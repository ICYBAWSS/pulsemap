//! On-disk embedding cache keyed by file fingerprint (name|size|mtime), the
//! native replacement for the browser's IndexedDB cache. One JSON file loaded
//! at startup and rewritten on change — simple and robust for the library sizes
//! this tool handles.

use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct CacheEntry {
    pub path: String,
    pub filename: String,
    pub embedding: Vec<f32>,
    pub section: String,
    pub confidence: f32,
    pub tags: Vec<(String, f32)>,
    /// Playback waveform data. `default` so a pre-v3 cache file still loads.
    #[serde(default)]
    pub envelope: Vec<f32>,
    #[serde(default)]
    pub duration: f32,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Cache {
    #[serde(skip)]
    file: PathBuf,
    entries: HashMap<String, CacheEntry>,
}

/// Bump when the embedding pipeline changes so stale entries auto-invalidate.
/// v2: added silence-trim before mel (v1 embedded untrimmed → wrong space).
/// v3: entries now carry the waveform envelope + duration.
/// v4: relative-margin classification — cached section/confidence are stale.
/// v5: Unsorted threshold retuned (0.06 -> 0.015).
/// v6: linear classifier head + Crash/Ride merged into Cymbal.
const PIPE_VERSION: u32 = 6;

/// Stable per-file key: `vN|name|size|mtime` (mirrors the browser fingerprint,
/// plus a pipeline version so a preprocessing change busts the whole cache).
pub fn fingerprint(path: &Path) -> Result<String, Box<dyn Error>> {
    let meta = std::fs::metadata(path)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let mtime = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    Ok(format!("v{PIPE_VERSION}|{name}|{}|{mtime}", meta.len()))
}

impl Cache {
    /// Load (or start empty) a cache file.
    pub fn load(file: PathBuf) -> Self {
        let entries = std::fs::read_to_string(&file)
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, CacheEntry>>(&s).ok())
            .unwrap_or_default();
        Cache { file, entries }
    }

    pub fn get(&self, fp: &str) -> Option<&CacheEntry> {
        self.entries.get(fp)
    }

    /// File path -> embedding, for re-placing a reclassified node among its new
    /// siblings: the map holds only paths, and the fingerprint keys here can't
    /// be recomputed from one without touching the disk.
    pub fn embeddings_by_path(&self) -> HashMap<&str, &[f32]> {
        self.entries
            .values()
            .map(|e| (e.path.as_str(), e.embedding.as_slice()))
            .collect()
    }

    pub fn insert(&mut self, fp: String, entry: CacheEntry) {
        self.entries.insert(fp, entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every entry and persist the now-empty cache — the "Clear cache"
    /// button, so the next folder drop re-analyzes from scratch. Returns how
    /// many entries were removed.
    pub fn clear(&mut self) -> Result<usize, Box<dyn Error>> {
        let n = self.entries.len();
        self.entries.clear();
        self.save()?;
        Ok(n)
    }

    pub fn save(&self) -> Result<(), Box<dyn Error>> {
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&self.file, serde_json::to_string(&self.entries)?)?;
        Ok(())
    }
}
