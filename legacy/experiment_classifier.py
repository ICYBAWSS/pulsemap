#!/usr/bin/env python3
"""
Cheap classifier tweaks, measured on the SAME leakage-safe harness as
memorization_audit.py so we don't fool ourselves. Two levers:

  1. prototypes-per-class sweep (does finer granularity help?)
  2. per-class radius normalization: score = dist_to_class / class_radius.
     Raw nearest-prototype Euclidean unfairly penalizes acoustically spread
     classes (Cymbal/metallic) vs tight ones (Kick) -- normalizing by each
     class's own intra-cluster spread should give fairer margins.

Reuses load()/near_dup_groups() from memorization_audit (the honest groups).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from memorization_audit import load, near_dup_groups, N_SPLITS


def cv(X, y, groups, k, normalize):
    yt, yp = [], []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        classes = np.unique(y[tr])
        protos, radius = {}, {}
        for c in classes:
            Xc = Xtr[y[tr] == c]
            kc = min(k, len(Xc))
            km = KMeans(n_clusters=kc, n_init=3, random_state=42).fit(Xc)
            protos[c] = km.cluster_centers_
            # class radius = RMS distance of its own points to nearest prototype
            dnn = ((Xc[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1).min(1)
            radius[c] = np.sqrt(dnn.mean()) + 1e-6
        # score each test point: nearest-prototype dist per class (optionally / radius)
        scores = np.empty((len(Xte), len(classes)))
        for ci, c in enumerate(classes):
            dmin = np.sqrt(((Xte[:, None, :] - protos[c][None]) ** 2).sum(-1).min(1))
            scores[:, ci] = dmin / radius[c] if normalize else dmin
        yt.append(y[te])
        yp.append(classes[scores.argmin(1)])
    return np.concatenate(yt), np.concatenate(yp)


def main():
    X, y, packs = load()
    print(f"{len(X)} samples, {len(set(y))} classes")
    groups, _, _ = near_dup_groups(X, packs)
    print(f"{len(set(groups))} leakage groups\n")

    print(f"{'config':28s} {'overall':>8s} {'balanced':>9s}")
    for k in (20, 40, 60, 80):
        for norm in (False, True):
            yt, yp = cv(X, y, groups, k, norm)
            acc = (yt == yp).mean()
            classes = sorted(set(y))
            bal = np.mean([(yp[yt == c] == c).mean() for c in classes if (yt == c).any()])
            tag = f"k={k:<3d} {'radius-norm' if norm else 'raw       '}"
            print(f"{tag:28s} {acc*100:7.2f}% {bal*100:8.2f}%")


if __name__ == "__main__":
    main()
