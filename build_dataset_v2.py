#!/usr/bin/env python3
"""Reproduce the BEST KNOWN dataset (see RESULTS.md): frozen CLAP embeddings with
(1) top_db=60 silence trim preserving decay, (2) filename-vs-folder label
cleanup, (3) _unsorted recovery. Writes training_data/embeddings_v2.npz.
A class-weighted LogisticRegression(C=0.3) on this, leakage-safe GroupKFold,
scores overall 74.7% / balanced 62.5% on the full 20-class taxonomy.

Usage: venv-clap/bin/python build_dataset_v2.py
"""
import warnings; warnings.filterwarnings("ignore")
import csv, re, os, glob, time, numpy as np, librosa, torch
from transformers import ClapModel, ClapProcessor
SR = 48000
TD = "training_data"

# Filename keyword -> label. Ordered: first match wins. Applied to the file name
# with bracketed tags [..]/(..) stripped first, because those hold pack/artist
# names (e.g. "[808 Mafia]") that would otherwise mis-trigger the sound match.
RULES = [
    ("808", r"808"), ("Kick", r"\bkick|\bbd\b|bass ?drum|\bbdrum"),
    ("Closed Hat", r"closed.?hat|closed.?hh|\bchh\b|\bchat\b|clsd|closed.?hi|closed.?cym"),
    ("Open Hat", r"open.?hat|open.?hh|\bohh\b|\bohat\b|open.?hi|open.?cym"),
    ("Rimshot", r"rim"), ("Crash", r"crash|china|splash"), ("Ride", r"\bride|sizzle"),
    ("Cymbal", r"cymbal|\bcym\b"), ("Cowbell", r"cowbell|cow.?bell"), ("Clap", r"\bclap|\bclp\b"),
    ("Snare", r"\bsnare|\bsnr\b|\bsd\b"), ("Tom", r"\btom\b|\btoms\b"), ("Shaker", r"shaker|\bshake"),
    ("Tambourine", r"tamb"), ("Conga", r"conga|bongo"), ("Snap", r"snap|finger"),
    ("Woodblock", r"wood.?block|\bclave"), ("Rolls", r"\broll(?!\s*(with|on|the|me|in|up))|buzz roll"),
    ("Vocal", r"vocal|\bvox\b|adlib|ad.?lib"), ("Bass", r"\bbass\b|\bsub\s?bass"),
]

def fnlabel(name):
    n = re.sub(r"[\[\(].*?[\]\)]", " ", name).lower()
    for lab, pat in RULES:
        if re.search(pat, n):
            return lab
    return None

def trim60(a):
    """Silence-trim at -60 dB (preserves cymbal/ride decay tails; -30 clipped them)."""
    peak = np.max(np.abs(a)); fr = 1024
    if peak == 0:
        return a
    th = peak * (10 ** (-60 / 20.0)); s = 0; e = len(a)
    for i in range(0, len(a), fr):
        if np.sqrt(np.mean(a[i:i+fr] ** 2)) > th: s = i; break
    for i in range(len(a), 0, -fr):
        if np.sqrt(np.mean(a[max(0, i-fr):i] ** 2)) > th: e = i; break
    return a[s:e] if s < e else a

def main():
    rows = list(csv.DictReader(open(f"{TD}/manifest.csv")))
    disk = {}
    for p in glob.glob(f"{TD}/labeled/**/*", recursive=True) + glob.glob(f"{TD}/unsorted/**/*", recursive=True):
        if p.lower().endswith((".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")):
            disk.setdefault(os.path.splitext(os.path.basename(p))[0][:12], p)
    items = []
    for r in rows:
        h = r["hash"][:12]; fol = r["label"]; fl = fnlabel(r["orig_name"])
        if h not in disk:
            continue
        if fol == "_unsorted":
            if fl:
                items.append((disk[h], fl))                    # recovered data
        else:
            items.append((disk[h], fl if (fl and fl != fol) else fol))  # correction overlay
    print(f"{len(items)} files to embed")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    X, Y, P = [], [], []; t0 = time.time()
    for i, (p, lab) in enumerate(items):
        try:
            a = trim60(librosa.load(p, sr=SR, mono=True)[0])
            if len(a) < 256: a = np.pad(a, (0, 256))
            inp = proc(audio=a, sampling_rate=SR, return_tensors="pt").to(dev)
            with torch.no_grad():
                e = torch.nn.functional.normalize(m.get_audio_features(**inp).pooler_output, dim=-1).cpu().numpy()[0]
            X.append(e.astype(np.float32)); Y.append(lab); P.append(p)
        except Exception:
            pass
        if i % 1000 == 0:
            print(f"\r {i+1}/{len(items)} ({(i+1)/(time.time()-t0):.0f}/s)", end="", flush=True)
    np.savez(f"{TD}/embeddings_v2.npz", X=np.array(X), y=np.array(Y), paths=np.array(P))
    print(f"\nDONE {len(X)} -> {TD}/embeddings_v2.npz")

if __name__ == "__main__":
    main()
