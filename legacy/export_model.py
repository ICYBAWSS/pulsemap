#!/usr/bin/env python3
"""
Export the production classifier as a linear head: StandardScaler params +
one weight matrix. Replaces export_prototypes.py's k-means nearest-prototype.

Why the change: nearest-prototype existed so the BROWSER could classify without
ML libraries (see export_prototypes.py's docstring). The native app has no such
constraint. Measured on leakage-safe pack-split CV over 14,526 sounds:

    head                        overall   balanced    size
    nearest-prototype (old)      76.56%     57.14%    3.9 MB
    logistic C=0.1 (this)        79.88%     60.48%     41 KB
    RBF SVM C=10                 81.08%     58.04%    13.3 MB

Balanced accuracy is the number that matters — the corpus is ~84:1 imbalanced,
and the rare classes are the ones users notice being wrong. The linear head wins
it outright, at 1/300th the size, as one matrix multiply, and it emits real
calibrated probabilities instead of a hand-rolled distance ratio.

Usage:  venv-clap/bin/python export_model.py
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMB = os.environ.get("EMB_CACHE", "training_data/embeddings.npz")
OUT = "model.json"
PRUNE_THRESH = 0.02
C = 0.1  # tuned on pack-split CV; higher overfits the big classes

# Merge the metallic family. Measured: "Cymbal" collapsed under evaluation —
# 233 real cymbals, predicted only 39 times, going to Open Hat 49% / Crash 26%.
# A crash IS a cymbal and a ride IS a cymbal, so the labels never named distinct
# sounds. Collapsing them lifted balanced accuracy 58.0% -> 62.1% and metallic
# accuracy 73.6% -> 76.7%, with non-metallic classes unchanged (83.7% -> 83.7%).
MERGE = {"Crash": "Cymbal", "Ride": "Cymbal"}


def flag_noise(X, y, labels):
    """Drop samples the model is confident are mislabeled (same as before)."""
    idx = {c: i for i, c in enumerate(labels)}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
        X, y, cv=cv, method="predict_proba")
    p_given = np.array([proba[i, idx[y[i]]] for i in range(len(y))])
    pred = np.array([labels[j] for j in proba.argmax(1)])
    return (p_given < PRUNE_THRESH) & (pred != y)


def main():
    d = np.load(EMB, allow_pickle=True)
    X, y = np.array(d["X"]), np.array(d["y"])
    for src, dst in MERGE.items():
        y[y == src] = dst
    labels = sorted(set(y))
    print(f"{len(X)} samples, {len(labels)} classes (after merging {MERGE})")

    flag = flag_noise(X, y, labels)
    Xc, yc = X[~flag], y[~flag]
    print(f"training on {len(Xc)} ({flag.sum()} pruned as likely mislabeled)")

    scaler = StandardScaler().fit(Xc)
    clf = LogisticRegression(max_iter=4000, C=C, class_weight="balanced").fit(
        scaler.transform(Xc), yc)

    out = {
        "labels": list(clf.classes_),
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
        "intercept": [round(float(v), 6) for v in clf.intercept_],
        "embedding_dim": int(Xc.shape[1]),
        "model": "Xenova/clap-htsat-unfused",
        "head": "multinomial logistic regression (softmax over scaled embedding)",
        "note": "scale the L2-normalized embedding, then softmax(coef @ x + intercept). "
                "Top probability picks the class; it is a real probability, so the "
                "Unsorted cutoff is a probability threshold.",
    }
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"wrote {len(clf.classes_)} classes x {Xc.shape[1]} dims -> {OUT} "
          f"({os.path.getsize(OUT)/1024:.0f} KB)")
    print(f"classes: {list(clf.classes_)}")


if __name__ == "__main__":
    main()
