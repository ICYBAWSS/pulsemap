#!/usr/bin/env python3
"""
Rank the metallic-family taxonomy options by measured quality.

"Cymbal" collapses under evaluation: 233 real Cymbals, predicted only 39 times,
going to Open Hat 49% / Crash 26%. A crash IS a cymbal and a ride IS a cymbal,
so the label may simply not name a distinct sound. This tests the alternatives.

CAVEAT built into the output: merging classes MECHANICALLY raises overall
accuracy (fewer ways to be wrong), so overall alone would always favour merging.
The honest comparisons are:
  - metallic-region accuracy: are metallic sounds filed trustworthily?
  - non-metallic accuracy: did we damage anything else? (should stay flat)

Usage:  venv-clap/bin/python taxonomy.py
"""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from memorization_audit import load, N_SPLITS

METAL = {"Cymbal", "Crash", "Ride", "Open Hat", "Closed Hat"}

VARIANTS = {
    "A. baseline (20 classes)": {},
    "B. Cymbal -> Crash": {"Cymbal": "Crash"},
    "C. Cymbal -> Open Hat": {"Cymbal": "Open Hat"},
    "D. Cymbal+Crash+Ride -> Cymbal": {"Crash": "Cymbal", "Ride": "Cymbal"},
    "E. drop Cymbal entirely": {"Cymbal": None},
}


def run(X, y, packs):
    yt, yp = [], []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, packs):
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced").fit(A, y[tr])
        yt.append(y[te])
        yp.append(clf.predict(B))
    return np.concatenate(yt), np.concatenate(yp)


def main():
    X0, y0, packs0 = load()
    print(f"{len(X0)} samples, RBF SVM, pack-split CV\n")
    print(f"{'variant':<32} {'classes':>8} {'overall':>9} {'balanced':>9} {'metallic':>9} {'other':>8}")
    for name, mapping in VARIANTS.items():
        y = y0.copy()
        keep = np.ones(len(y), dtype=bool)
        for src, dst in mapping.items():
            if dst is None:
                keep &= y != src
            else:
                y[y == src] = dst
        X, yy, packs = X0[keep], y[keep], packs0[keep]
        yt, yp = run(X, yy, packs)
        classes = sorted(set(yt))
        overall = (yt == yp).mean()
        bal = np.mean([(yp[yt == c] == c).mean() for c in classes])
        m = np.isin(yt, list(METAL))
        metal = (yt[m] == yp[m]).mean() if m.any() else float("nan")
        other = (yt[~m] == yp[~m]).mean()
        print(f"{name:<32} {len(classes):>8} {overall*100:>8.2f}% {bal*100:>8.2f}% "
              f"{metal*100:>8.2f}% {other*100:>7.2f}%", flush=True)


if __name__ == "__main__":
    main()
