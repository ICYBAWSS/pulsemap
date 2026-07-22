#!/usr/bin/env python3
"""
Extract CLAP audio embeddings for the whole labeled corpus, once, and cache them.

Embedding is the slow part (a forward pass per sound); training on top is
instant. So we do this separately and save to training_data/embeddings.npz.
Re-running is incremental: already-embedded files are skipped, so you can add
more kits with collect.py and just run this again.

Usage:  source venv-clap/bin/activate && python embed.py
"""
import os
import sys
import glob
import time
import warnings
import numpy as np
import librosa
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

MODEL_ID = os.environ.get("CLAP_MODEL", "laion/clap-htsat-unfused")
SR = 48000
LABELED_DIR = "training_data/labeled"
CACHE = os.environ.get("EMB_CACHE", "training_data/embeddings.npz")
BATCH = 32
LOAD_WORKERS = 8
SAVE_EVERY = 1000  # checkpoint the cache this often (files)


def list_labeled():
    """Return [(path, label), ...] for every audio file under LABELED_DIR."""
    out = []
    for label in sorted(os.listdir(LABELED_DIR)):
        d = os.path.join(LABELED_DIR, label)
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*")):
            if f.lower().endswith((".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")):
                out.append((f, label))
    return out


TRIM = os.environ.get("TRIM", "1") == "1"


def js_trim_silence(samples, top_db=30, frame_size=1024):
    n = len(samples)
    if n == 0:
        return samples
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples
    threshold = peak * (10 ** (-top_db / 20.0))
    
    start = 0
    end = n
    
    # Scan forward
    for i in range(0, n, frame_size):
        chunk = samples[i:min(i + frame_size, n)]
        if len(chunk) == 0:
            continue
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > threshold:
            start = i
            break
            
    # Scan backward
    for i in range(n, 0, -frame_size):
        s = max(0, i - frame_size)
        chunk = samples[s:i]
        if len(chunk) == 0:
            continue
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > threshold:
            end = i
            break
            
    return samples[start:end] if start < end else samples


def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        if TRIM:
            y = js_trim_silence(y, top_db=30)
        if len(y) < 256:
            return None
        return y.astype(np.float32)
    except Exception:
        return None


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    items = list_labeled()
    print(f"Found {len(items)} labeled files.")

    # Resume: load existing cache and skip already-embedded paths.
    done_paths, X_old, y_old, p_old = set(), [], [], []
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        X_old, y_old, p_old = list(d["X"]), list(d["y"]), list(d["paths"])
        done_paths = set(p_old)
        print(f"Cache has {len(done_paths)} embeddings already — skipping those.")

    todo = [(p, l) for (p, l) in items if p not in done_paths]
    print(f"Embedding {len(todo)} new files on {device} (batch={BATCH})...")
    if not todo:
        print("Nothing to do. Cache is up to date.")
        return

    print(f"Loading CLAP ({MODEL_ID})...")
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)

    X, y, paths = X_old, y_old, p_old
    t0 = time.time()
    n_new = 0

    def flush():
        np.savez(CACHE, X=np.array(X), y=np.array(y), paths=np.array(paths))

    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as pool:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            audios = list(pool.map(load_audio, [p for p, _ in chunk]))
            keep = [(a, lbl, p) for a, (p, lbl) in zip(audios, chunk) if a is not None]
            if not keep:
                continue
            with torch.no_grad():
                embs_chunk = []
                for a, _, _ in keep:
                    # Process one audio at a time to avoid padding and match the browser exactly
                    inp = proc(audio=a, sampling_rate=SR, return_tensors="pt").to(device)
                    emb = model.get_audio_features(**inp).pooler_output
                    emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()[0]
                    embs_chunk.append(emb)
                emb = np.array(embs_chunk)
            for vec, (_, lbl, p) in zip(emb, keep):
                X.append(vec.astype(np.float32)); y.append(lbl); paths.append(p)
                n_new += 1

            if n_new % SAVE_EVERY < BATCH:
                flush()
            done = i + len(chunk)
            rate = done / (time.time() - t0 + 1e-9)
            eta = (len(todo) - done) / (rate + 1e-9)
            print(f"\r  {done}/{len(todo)}  ({rate:.1f} files/s, ETA {eta/60:.1f} min)   ",
                  end="", flush=True)

    flush()
    print(f"\nDone. Cache now holds {len(X)} embeddings at {CACHE}.")


if __name__ == "__main__":
    main()
