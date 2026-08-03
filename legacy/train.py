#!/usr/bin/env python3
"""
Train the classifier head on cached CLAP embeddings and report honest accuracy.

Reads training_data/embeddings.npz (produced by embed.py), trains a
logistic-regression probe with stratified cross-validation, and prints overall +
per-class accuracy and the worst confusions. Classes are weighted to handle the
big imbalance (Snare 3000+ vs Woodblock ~50).

Usage:  source venv-clap/bin/activate && python train.py
"""
import os
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CACHE = "training_data/embeddings.npz"
MODEL_OUT = "training_data/probe.joblib"
MIN_PER_CLASS = 20   # classes below this are too small to trust; reported but flagged


def main():
    if not os.path.exists(CACHE):
        print(f"No embeddings at {CACHE}. Run embed.py first.")
        return
    d = np.load(CACHE, allow_pickle=True)
    X, y = np.array(d["X"]), np.array(d["y"])
    print(f"Loaded {len(X)} embeddings, {len(set(y))} classes.")

    counts = Counter(y)
    small = [c for c, n in counts.items() if n < MIN_PER_CLASS]
    if small:
        print(f"  (small classes, treat their numbers with caution: "
              f"{', '.join(f'{c}={counts[c]}' for c in small)})")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
    )

    # Cross-validated predictions for an honest accuracy estimate.
    n_splits = min(5, min(counts.values()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(f"\nRunning {n_splits}-fold cross-validation...")
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    acc = (pred == y).mean()
    # Balanced accuracy = mean per-class recall (fairer under imbalance).
    labels = sorted(counts.keys())
    per_class_recall = {
        c: (pred[y == c] == c).mean() for c in labels
    }
    bal_acc = np.mean(list(per_class_recall.values()))

    print(f"\n=== Overall accuracy: {acc*100:.1f}%   "
          f"Balanced (mean per-class): {bal_acc*100:.1f}% ===\n")

    print("Per-class recall (n):")
    for c in sorted(labels, key=lambda k: per_class_recall[k]):
        flag = "  <-- few samples" if counts[c] < MIN_PER_CLASS else ""
        print(f"  {c:12s} {per_class_recall[c]*100:5.1f}%  (n={counts[c]}){flag}")

    # Top confusions.
    cm = confusion_matrix(y, pred, labels=labels)
    print("\nTop confusions (true -> predicted, count):")
    conf = []
    for i, tc in enumerate(labels):
        for j, pc in enumerate(labels):
            if i != j and cm[i, j] > 0:
                conf.append((cm[i, j], tc, pc))
    for cnt, tc, pc in sorted(conf, reverse=True)[:12]:
        print(f"  {tc:12s} -> {pc:12s} {cnt}")

    # Fit final model on ALL data and save it for the pipeline to use.
    clf.fit(X, y)
    joblib.dump({"clf": clf, "labels": labels}, MODEL_OUT)
    print(f"\nSaved trained probe to {MODEL_OUT}")


if __name__ == "__main__":
    main()
