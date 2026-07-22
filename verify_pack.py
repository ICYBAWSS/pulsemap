#!/usr/bin/env python3
"""
Verify the shipped classifier (prototypes.json) against a specific source
pack's folder-name ground truth. Folder/file names in a pack are a free,
reliable label signal (e.g. "Kick/JIRO_FUNKY_BD_*.wav") -- use them to check
where the shipped model actually gets it wrong, instead of guessing.

Usage:  source venv-clap/bin/activate && python verify_pack.py "<source_folder>"
"""
import sys
import csv
import json
import numpy as np
from collections import Counter, namedtuple

EMB = "training_data/embeddings_final.npz"
MANIFEST = "training_data/manifest.csv"
PROTOTYPES = "prototypes.json"


def load_classifier():
    with open(PROTOTYPES) as f:
        p = json.load(f)
    mean = np.array(p["scaler_mean"])
    scale = np.array(p["scaler_scale"])
    protos = np.array([pr["vec"] for pr in p["prototypes"]])
    proto_labels = np.array([pr["label"] for pr in p["prototypes"]])
    return mean, scale, protos, proto_labels


def classify(X, mean, scale, protos, proto_labels):
    """Faithful port of native/src/classify.rs: per-class nearest-prototype
    distance, then margin = (2nd-best class dist) - (best class dist), ABSOLUTE."""
    Xs = (X - mean) / scale
    d = np.linalg.norm(Xs[:, None, :] - protos[None, :, :], axis=2)  # (n, n_protos)
    classes = sorted(set(proto_labels))
    # nearest prototype distance per class
    per_class = np.stack(
        [d[:, proto_labels == c].min(axis=1) for c in classes], axis=1)  # (n, n_classes)
    order = np.argsort(per_class, axis=1)
    best = np.array(classes)[order[:, 0]]
    best_d = per_class[np.arange(len(X)), order[:, 0]]
    second_d = per_class[np.arange(len(X)), order[:, 1]]
    margin = second_d - best_d  # absolute, matches classify.rs
    return best, margin


def main():
    pack = sys.argv[1] if len(sys.argv) > 1 else "Jiro Inagaki & Soul Media - Funky Stuff Pack"

    Row = namedtuple("Row", ["hash", "label", "duration", "orig_name", "source_folder", "source_link"])
    with open(MANIFEST, newline="") as f:
        rows = [Row(**r) for r in csv.DictReader(f)]
    man = [r for r in rows if r.source_folder == pack]
    if not man:
        print(f"No manifest rows for source_folder={pack!r}")
        return
    hash_to_row = {row.hash[:12]: row for row in man}

    d = np.load(EMB, allow_pickle=True)
    X, y, paths = np.array(d["X"]), np.array(d["y"]), np.array(d["paths"])
    # paths look like training_data/labeled/<Label>/<hash><ext>
    hashes = np.array([p.split("/")[-1].split(".")[0] for p in paths])
    mask = np.isin(hashes, list(hash_to_row.keys()))
    Xp, yp, hp = X[mask], y[mask], hashes[mask]
    print(f"{pack}: {len(Xp)} embedded files found (of {len(man)} in manifest)")

    mean, scale, protos, proto_labels = load_classifier()
    pred, margin = classify(Xp, mean, scale, protos, proto_labels)

    acc = (pred == yp).mean()
    print(f"\nOverall accuracy vs folder ground truth: {acc*100:.1f}%\n")

    labels = sorted(set(yp))
    print(f"{'label':12s} {'n':>4s} {'acc':>6s}")
    for c in labels:
        m = yp == c
        if m.sum():
            print(f"{c:12s} {m.sum():4d} {(pred[m] == c).mean()*100:5.1f}%")

    print("\nConfusion (true -> predicted, count):")
    conf = Counter(zip(yp, pred))
    for (t, p), n in sorted(conf.items(), key=lambda kv: -kv[1]):
        if t != p:
            print(f"  {t:12s} -> {p:12s}  x{n}")

    MARGIN_UNSORTED = 0.20  # matches native/src/classify.rs, pipeline.js
    uns = np.where(margin < MARGIN_UNSORTED)[0]
    uns = uns[np.argsort(margin[uns])]
    print(f"\n'Unsorted' pile: {len(uns)}/{len(Xp)} files fall below margin {MARGIN_UNSORTED}")
    print("(true_label -> top guess, margin) — these are what the app declines to commit:")
    for i in uns:
        row = hash_to_row[hp[i]]
        ok = "correct" if pred[i] == yp[i] else "WRONG  "
        print(f"  [{yp[i]:10s} -> {pred[i]:10s} m={margin[i]:.2f} {ok}]  {row.orig_name}")

    print("\nUnsorted by true class:")
    for c in labels:
        m = (margin < MARGIN_UNSORTED) & (yp == c)
        if m.sum():
            print(f"  {c:12s} {m.sum():3d}/{(yp==c).sum()}")


if __name__ == "__main__":
    main()
