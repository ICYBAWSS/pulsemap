#!/usr/bin/env python3
"""
How much accuracy is the nearest-prototype classifier costing us?

It was picked so the browser could run it without ML libraries (see
export_prototypes.py). The native app has no such constraint — any of these
heads is a short matrix routine in Rust. This compares them on the SAME honest
pack-split CV, reporting balanced accuracy too, because the corpus is ~84:1
imbalanced and the weak classes are what actually annoys users.

Usage:  venv-clap/bin/python heads.py
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from memorization_audit import load, N_SPLITS, PROTOTYPES_PER_CLASS


def prototype_head(A, ytr, B):
    classes = np.unique(ytr)
    P, lab = [], []
    for c in classes:
        Xc = A[ytr == c]
        k = min(PROTOTYPES_PER_CLASS, len(Xc))
        km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(Xc)
        P.append(km.cluster_centers_)
        lab += [c] * k
    P = np.vstack(P).astype(np.float32)
    lab = np.array(lab)
    pn = (P * P).sum(1)
    out = np.empty(len(B), dtype=np.int64)
    for s in range(0, len(B), 1024):
        blk = B[s : s + 1024]
        d2 = (blk * blk).sum(1)[:, None] + pn[None, :] - 2.0 * (blk @ P.T)
        out[s : s + 1024] = d2.argmin(1)
    return lab[out]


HEADS = {
    "nearest-prototype (current)": prototype_head,
    "logistic (balanced)": lambda A, y, B: LogisticRegression(
        max_iter=3000, C=1.0, class_weight="balanced", n_jobs=-1).fit(A, y).predict(B),
    "linear SVM (balanced)": lambda A, y, B: SVC(
        kernel="linear", C=1.0, class_weight="balanced").fit(A, y).predict(B),
    "RBF SVM (balanced)": lambda A, y, B: SVC(
        kernel="rbf", C=10.0, gamma="scale", class_weight="balanced").fit(A, y).predict(B),
    "MLP (256 hidden)": lambda A, y, B: MLPClassifier(
        hidden_layer_sizes=(256,), max_iter=400, random_state=42).fit(A, y).predict(B),
}


def main():
    X, y, packs = load()
    print(f"{len(X)} samples, {len(set(y))} classes, {len(set(packs))} packs")
    print("pack-split CV (leakage-safe)\n")
    classes = sorted(set(y))
    print(f"{'head':<30} {'overall':>9} {'balanced':>10}")
    results = {}
    for name, fn in HEADS.items():
        yt, yp = [], []
        for tr, te in GroupKFold(N_SPLITS).split(X, y, packs):
            sc = StandardScaler().fit(X[tr])
            A, B = sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
            yt.append(y[te])
            yp.append(fn(A, y[tr], B))
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        overall = (yt == yp).mean()
        bal = np.mean([(yp[yt == c] == c).mean() for c in classes if (yt == c).any()])
        results[name] = (yt, yp)
        print(f"{name:<30} {overall*100:>8.2f}% {bal*100:>9.2f}%", flush=True)

    # Per-class, best vs current — where does the gain actually land?
    base = results["nearest-prototype (current)"]
    best = max(results.items(), key=lambda kv: (kv[1][0] == kv[1][1]).mean())
    print(f"\nper-class: current vs {best[0]}")
    yt_b, yp_b = base
    yt_n, yp_n = best[1]
    print(f"{'class':<12} {'n':>5} {'current':>9} {'best':>9} {'delta':>8}")
    for c in classes:
        m = yt_b == c
        if not m.any():
            continue
        a = (yp_b[m] == c).mean() * 100
        d = (yp_n[yt_n == c] == c).mean() * 100
        print(f"{c:<12} {m.sum():>5} {a:>8.1f}% {d:>8.1f}% {d-a:>+7.1f}")


if __name__ == "__main__":
    main()
