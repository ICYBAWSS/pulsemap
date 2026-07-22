#!/usr/bin/env python3
"""
Build the sectioned canvas for the end user.

Takes a folder of sounds (the thing a producer drops in), and for each one-shot:
  1. embeds it with CLAP
  2. classifies it into a named SECTION via the trained probe (probe.joblib),
     with a confidence and the top-3 soft tags (for search)
  3. lays out each section's sounds locally with UMAP
  4. arranges the sections on a global canvas so acoustically-similar families
     sit next to each other
Low-confidence sounds go to an "Unsorted" island so nothing is ever hidden.

Output: data.json consumed by the front-end (app.js).

Usage:  source venv-clap/bin/activate && python build_map.py [folder]
"""
import os
import sys
import json
import glob
import warnings
import numpy as np
import librosa
import torch
import joblib
import umap
from concurrent.futures import ThreadPoolExecutor
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "test_samples"
OUT = "data.json"
MODEL_ID = "laion/clap-htsat-unfused"
PROBE = "training_data/probe.joblib"
SR = 48000
MAX_ONESHOT_SEC = 3.0
# "Unsorted" when the classifier isn't clearly deciding: the gap between its top
# and second choice (decision margin) is small. More meaningful than softmax,
# which SVC compresses to a near-constant.
MARGIN_UNSORTED = 0.9
AUDIO_EXTS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")

# Fixed, semantic-ish colors so a section always looks the same. Grouped by
# family (lows warm, snares/claps pink, metals cyan/blue, perc greens, vocal gold).
COLORS = {
    "Kick": "#ff6b4a", "808": "#ff4d6d", "Snare": "#f65fb0", "Clap": "#e07bff",
    "Rimshot": "#c98bff", "Snap": "#a78bfa", "Closed Hat": "#4dd0e1",
    "Open Hat": "#4aa8ff", "Ride": "#5c9dff", "Crash": "#6d8bff", "Cymbal": "#7aa0ff",
    "Tom": "#ffb04a", "Conga": "#ffc857", "Cowbell": "#ffd94a",
    "Shaker": "#8bd94a", "Tambourine": "#4ad98b", "Woodblock": "#4ad9c0",
    "Vocal": "#ffe14a", "Unsorted": "#5a6472",
}


def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
        return y.astype(np.float32) if len(y) >= 256 else None
    except Exception:
        return None


def embed_folder(files, device):
    print(f"Embedding {len(files)} sounds...")
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    embs, keep = [], []
    B = 32
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(files), B):
            chunk = files[i:i + B]
            audios = list(pool.map(load_audio, chunk))
            pairs = [(a, f) for a, f in zip(audios, chunk) if a is not None]
            if not pairs:
                continue
            with torch.no_grad():
                inp = proc(audio=[a for a, _ in pairs], sampling_rate=SR,
                           return_tensors="pt", padding=True).to(device)
                e = model.get_audio_features(**inp).pooler_output
                e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
            embs.extend(e.astype(np.float32))
            keep.extend(f for _, f in pairs)
            print(f"\r  {min(i+B,len(files))}/{len(files)}", end="", flush=True)
    print()
    return np.array(embs), keep


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def local_layout(emb):
    """2D coords in a unit box for one section's sounds."""
    n = len(emb)
    if n == 1:
        return np.array([[0.5, 0.5]])
    if n <= 6:  # too few for UMAP -> circle
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([0.5 + 0.35 * np.cos(ang), 0.5 + 0.35 * np.sin(ang)], 1)
    reducer = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.12,
                        n_components=2, random_state=42)
    xy = reducer.fit_transform(emb)
    xy -= xy.min(0)
    span = xy.max(0)
    span[span == 0] = 1
    return xy / span


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    saved = joblib.load(PROBE)
    model, labels = saved["model"], saved["labels"]

    files = [f for f in glob.glob(f"{FOLDER}/**/*", recursive=True)
             if f.lower().endswith(AUDIO_EXTS)]
    # keep only one-shots (loops are a separate experience)
    oneshots = []
    for f in files:
        try:
            info = librosa.get_duration(path=f)
            if info <= MAX_ONESHOT_SEC:
                oneshots.append(f)
        except Exception:
            pass
    print(f"{len(oneshots)} one-shots of {len(files)} files under {FOLDER}/")

    emb, kept = embed_folder(oneshots, device)
    if len(emb) == 0:
        print("Nothing to embed."); return

    # Classify: section + confidence + top-3 soft tags.
    scores = model.decision_function(emb)          # (n, n_classes)
    probs = softmax(scores)                         # for readable tag weights
    top = np.argsort(-scores, axis=1)[:, :3]
    pred_idx = top[:, 0]
    srt = np.sort(scores, axis=1)                   # ascending
    margin = srt[:, -1] - srt[:, -2]                # top choice minus runner-up
    conf = 1.0 / (1.0 + np.exp(-margin))            # squash margin -> 0..1 for display
    sections = [labels[i] if margin[k] >= MARGIN_UNSORTED else "Unsorted"
                for k, i in enumerate(pred_idx)]
    print(f"{sum(s == 'Unsorted' for s in sections)} of {len(sections)} -> Unsorted "
          f"(margin < {MARGIN_UNSORTED})")

    # Local layout per section.
    order = [l for l in labels if l in set(sections)]
    if "Unsorted" in sections:
        order.append("Unsorted")
    local = {}
    sec_centroid = {}
    for sec in order:
        idxs = [k for k, s in enumerate(sections) if s == sec]
        local[sec] = (idxs, local_layout(emb[idxs]))
        sec_centroid[sec] = emb[idxs].mean(0)

    # Global arrangement: a GRID of section tiles (robust for a handful of
    # sections, unlike UMAP which collapses with so few points). Sections are
    # ORDERED by acoustic similarity via a 1-D PCA projection of their centroids,
    # so kick/808 and hats/cymbals end up as neighbors in reading order. Unsorted
    # is parked last.
    secs = [s for s in order if s != "Unsorted"]
    cents = np.array([sec_centroid[s] for s in secs])
    cents_c = cents - cents.mean(0)
    pc1 = np.linalg.svd(cents_c, full_matrices=False)[2][0]
    secs = [s for _, s in sorted(zip(cents_c @ pc1, secs))]
    if "Unsorted" in order:
        secs.append("Unsorted")

    PAD = 22.0                 # size of each section's local map
    GAP = 14.0                 # gap between tiles
    CELL = PAD + GAP
    cols = int(np.ceil(np.sqrt(len(secs))))
    sec_origin = {s: np.array([(i % cols) * CELL, (i // cols) * CELL])
                  for i, s in enumerate(secs)}

    # Build node records with global coords.
    nodes = []
    for sec in secs:
        idxs, xy = local[sec]
        ox, oy = sec_origin[sec]
        for j, k in enumerate(idxs):
            gx = ox + xy[j, 0] * PAD
            gy = oy + xy[j, 1] * PAD
            tags = [[labels[t], round(float(probs[k, t]), 3)] for t in top[k]]
            nodes.append({
                "path": kept[k],
                "filename": os.path.basename(kept[k]),
                "section": sec,
                "confidence": round(float(conf[k]), 3),
                "tags": tags,
                "x": round(float(gx), 3),
                "y": round(float(gy), 3),
                "color": COLORS.get(sec, "#888888"),
            })

    # Section metadata for labels/legend/navigation.
    sec_meta = []
    for sec in secs:
        pts = np.array([[n["x"], n["y"]] for n in nodes if n["section"] == sec])
        sec_meta.append({
            "name": sec,
            "count": int(len(pts)),
            "color": COLORS.get(sec, "#888888"),
            "cx": round(float(pts[:, 0].mean()), 3),
            "cy": round(float(pts[:, 1].mean()), 3),
        })
    sec_meta.sort(key=lambda s: -s["count"])

    json.dump({"nodes": nodes, "sections": sec_meta}, open(OUT, "w"))
    print(f"\nWrote {len(nodes)} sounds across {len(secs)} sections -> {OUT}")
    for s in sec_meta:
        print(f"  {s['name']:12s} {s['count']}")


if __name__ == "__main__":
    main()
