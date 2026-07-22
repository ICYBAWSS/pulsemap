#!/usr/bin/env python3
"""
Export the production classifier as a small JSON file the browser can use with
zero ML libraries: just per-class k-means prototypes + StandardScaler params.

Classification in JS becomes: scale the embedding, find the nearest prototype
(Euclidean), and its class wins. Confidence = margin between the best and
second-best class's nearest distance (mirrors the Unsorted logic in build_map.py).

Usage:  source venv-clap/bin/activate && python export_prototypes.py
"""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import os
EMB = os.environ.get("EMB_CACHE", "training_data/embeddings.npz")
OUT = "prototypes.json"
PRUNE_THRESH = 0.02
PROTOTYPES_PER_CLASS = 40


def flag_noise(X, y, labels):
    idx = {c: i for i, c in enumerate(labels)}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
        X, y, cv=cv, method="predict_proba", n_jobs=-1)
    p_given = np.array([proba[i, idx[y[i]]] for i in range(len(y))])
    pred = np.array([labels[j] for j in proba.argmax(1)])
    return (p_given < PRUNE_THRESH) & (pred != y)


def main():
    d = np.load(EMB, allow_pickle=True)
    X, y, paths = np.array(d["X"]), np.array(d["y"]), np.array(d["paths"])
    labels = sorted(set(y))

    flag = flag_noise(X, y, labels)
    keep = ~flag
    Xc, yc = X[keep], y[keep]
    print(f"Training on {len(Xc)} cleaned samples ({flag.sum()} pruned as likely mislabeled).")

    scaler = StandardScaler().fit(Xc)
    Xs = scaler.transform(Xc)

    prototypes = []  # list of {label, vec}
    for c in labels:
        Xcls = Xs[yc == c]
        n_clusters = min(PROTOTYPES_PER_CLASS, len(Xcls))
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42).fit(Xcls)
        for center in km.cluster_centers_:
            prototypes.append({"label": c, "vec": [round(float(v), 5) for v in center]})

    out = {
        "labels": labels,
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "prototypes": prototypes,
        "embedding_dim": Xc.shape[1],
        "model": "Xenova/clap-htsat-unfused",
        "note": "Nearest-prototype classifier. Distance = Euclidean in scaled space. "
                "confidence/Unsorted = margin between best and 2nd-best class's nearest prototype.",
    }
    with open(OUT, "w") as f:
        json.dump(out, f)

    import os
    size_kb = os.path.getsize(OUT) / 1024
    print(f"Wrote {len(prototypes)} prototypes across {len(labels)} classes -> {OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
