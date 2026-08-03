#!/usr/bin/env python3
"""
Honest evaluation of the nearest-prototype classifier — does it GENERALIZE or
just MEMORIZE sounds?

The trick: sample packs contain near-duplicate / re-processed versions of the
same sound. If a pack straddles the train/test split, the model "remembers" and
accuracy is inflated. So we compare two evaluations of the SAME classifier:

  1. Random split (StratifiedKFold)  — same-pack leakage allowed (optimistic).
  2. Grouped split (GroupKFold by source_folder) — every test pack is UNSEEN
     in training. This is the honest generalization number.

The GAP between them is the memorization effect. We also run a 1-NN classifier
(pure memorization) as a reference point.

Usage:  venv-clap/bin/python backtest.py
"""
import csv
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

EMB = "training_data/embeddings_final.npz"
MANIFEST = "training_data/manifest.csv"
PROTOTYPES_PER_CLASS = 40
N_SPLITS = 5


def load():
    d = np.load(EMB, allow_pickle=True)
    X, y, paths = np.array(d["X"]), np.array(d["y"]), np.array(d["paths"])
    # Join source pack by hash prefix (path stem = first 12 chars of md5).
    src_by_hash = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            src_by_hash[r["hash"][:12]] = r["source_folder"]
    groups = np.array([
        src_by_hash.get(os.path.splitext(os.path.basename(str(p)))[0], "UNKNOWN")
        for p in paths
    ])
    return X, y, groups


def prototype_fit_predict(Xtr, ytr, Xte):
    """The production classifier: StandardScaler + per-class k-means prototypes,
    predict by nearest prototype (Euclidean) in scaled space."""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    proto_vecs, proto_lab = [], []
    for c in np.unique(ytr):
        Xc = Xtr_s[ytr == c]
        k = min(PROTOTYPES_PER_CLASS, len(Xc))
        km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(Xc)
        proto_vecs.append(km.cluster_centers_)
        proto_lab += [c] * k
    P = np.vstack(proto_vecs)
    proto_lab = np.array(proto_lab)
    # nearest prototype: argmin ||x - p||^2
    d2 = ((Xte_s[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return proto_lab[d2.argmin(1)]


def evaluate(splitter, X, y, groups, classifier, use_groups):
    y_true_all, y_pred_all = [], []
    it = splitter.split(X, y, groups) if use_groups else splitter.split(X, y)
    for tr, te in it:
        if classifier == "proto":
            pred = prototype_fit_predict(X[tr], y[tr], X[te])
        else:  # 1-NN on scaled embeddings (pure memorization baseline)
            sc = StandardScaler().fit(X[tr])
            knn = KNeighborsClassifier(1).fit(sc.transform(X[tr]), y[tr])
            pred = knn.predict(sc.transform(X[te]))
        y_true_all.append(y[te])
        y_pred_all.append(pred)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def report(tag, yt, yp):
    acc = (yt == yp).mean()
    print(f"  {tag:32s} accuracy = {acc*100:5.2f}%")
    return acc


def per_class_and_confusions(yt, yp):
    labels = sorted(set(yt))
    print("\n  per-class accuracy (honest / grouped split):")
    for c in labels:
        m = yt == c
        acc = (yp[m] == c).mean() if m.sum() else 0.0
        print(f"    {c:14s} {acc*100:5.1f}%   (n={m.sum()})")
    # top confusions
    from collections import Counter
    conf = Counter()
    for t, p in zip(yt, yp):
        if t != p:
            conf[(t, p)] += 1
    print("\n  top confusions (true -> predicted):")
    for (t, p), n in conf.most_common(10):
        print(f"    {t:14s} -> {p:14s} {n}")


def main():
    X, y, groups = load()
    print(f"{len(X)} samples, {len(set(y))} classes, {len(set(groups))} source packs\n")

    print("PROTOTYPE classifier (production):")
    yt_r, yp_r = evaluate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=42),
                          X, y, groups, "proto", use_groups=False)
    acc_random = report("random split (leaky/optimistic)", yt_r, yp_r)
    yt_g, yp_g = evaluate(GroupKFold(N_SPLITS), X, y, groups, "proto", use_groups=True)
    acc_group = report("grouped split (honest)", yt_g, yp_g)

    print("\n1-NN classifier (pure memorization reference):")
    knn_r = evaluate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=42),
                     X, y, groups, "knn", use_groups=False)
    report("random split", *knn_r)
    knn_g = evaluate(GroupKFold(N_SPLITS), X, y, groups, "knn", use_groups=True)
    report("grouped split", *knn_g)

    print(f"\n>>> MEMORIZATION GAP (prototype): {(acc_random-acc_group)*100:.2f} "
          f"percentage points (random {acc_random*100:.1f}% - grouped {acc_group*100:.1f}%)")
    print(">>> The grouped number is the real-world accuracy on unseen packs.")

    per_class_and_confusions(yt_g, yp_g)


if __name__ == "__main__":
    main()
