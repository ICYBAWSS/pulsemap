#!/usr/bin/env python3
"""
Validate a nearest-prototype classifier (k-means centroids per class) as a
JS-portable replacement for the trained sklearn SVC.

Why: porting sklearn's exact RBF-kernel one-vs-one decision_function to
JavaScript is intricate and easy to get subtly wrong. Nearest-prototype is
just Euclidean/cosine distance to a small set of per-class centroids -- trivial
to implement correctly in plain JS, and ships as a small JSON file instead of
raw support vectors.

This script measures the accuracy cost of that simplification via the same
cross-validation methodology as train_final.py, on the same cleaned corpus.
"""
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

EMB = "training_data/embeddings.npz"
PRUNE_THRESH = 0.02
PROTOTYPES_PER_CLASS = [10, 20, 40]


def flag_noise(X, y, labels):
    from sklearn.model_selection import cross_val_predict
    idx = {c: i for i, c in enumerate(labels)}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
        X, y, cv=cv, method="predict_proba", n_jobs=-1)
    p_given = np.array([proba[i, idx[y[i]]] for i in range(len(y))])
    pred = np.array([labels[j] for j in proba.argmax(1)])
    return (p_given < PRUNE_THRESH) & (pred != y)


def eval_prototypes(Xtr, ytr, Xte, yte, labels, k):
    """Build k-means prototypes per class on train fold, classify test fold by nearest prototype."""
    protos, proto_labels = [], []
    for c in labels:
        Xc = Xtr[ytr == c]
        n_clusters = min(k, len(Xc))
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42).fit(Xc)
        protos.append(km.cluster_centers_)
        proto_labels += [c] * n_clusters
    P = np.vstack(protos)
    proto_labels = np.array(proto_labels)

    # Nearest prototype by Euclidean distance (scaled space).
    d = np.linalg.norm(Xte[:, None, :] - P[None, :, :], axis=2)  # (n_test, n_protos)
    nearest = d.argmin(axis=1)
    pred = proto_labels[nearest]
    acc = (pred == yte).mean()
    bal = np.mean([(pred[yte == c] == c).mean() for c in labels if (yte == c).any()])
    return acc, bal


def main():
    d = np.load(EMB, allow_pickle=True)
    X, y, paths = np.array(d["X"]), np.array(d["y"]), np.array(d["paths"])
    labels = sorted(set(y))

    flag = flag_noise(X, y, labels)
    keep = ~flag
    Xc, yc = X[keep], y[keep]
    print(f"Cleaned corpus: {len(Xc)} samples, {len(labels)} classes\n")

    scaler = StandardScaler().fit(Xc)
    Xs = scaler.transform(Xc)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for k in PROTOTYPES_PER_CLASS:
        accs, bals = [], []
        for tr_idx, te_idx in cv.split(Xs, yc):
            a, b = eval_prototypes(Xs[tr_idx], yc[tr_idx], Xs[te_idx], yc[te_idx], labels, k)
            accs.append(a); bals.append(b)
        print(f"prototypes/class={k:3d}  overall {np.mean(accs)*100:5.1f}%  "
              f"balanced {np.mean(bals)*100:5.1f}%   (export size ~{k*len(labels)*512*4/1e6:.2f}MB raw)")

    print("\nFor reference, train_final.py's SVC-RBF on this same cleaned set: "
          "overall 92.1%  balanced 88.2%")


if __name__ == "__main__":
    main()
