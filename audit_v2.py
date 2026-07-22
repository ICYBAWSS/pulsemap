#!/usr/bin/env python3
"""
Corrected memorization audit.

The previous protocol (memorization_audit.py) built "leakage groups" by
union-find over (same pack) OR (cosine >= T), then ran GroupKFold on them.
Transitive closure wrecks that: A~B and B~C merges A with C even when A and C
are unrelated, so with 10.7k cross-pack near-dup pairs the whole corpus collapses
into one component (78.9% of all samples at T=0.995, 98.7% at T=0.90). That
giant group has to land entirely in one fold, so that fold trains on a tiny
sliver and tests on everything -- the resulting "accuracy" mostly measures
data starvation, not memorization.

This version keeps folds balanced and still removes twins:
  1. GroupKFold over PACKS only (329 packs -> genuinely balanced folds).
  2. For each fold, DROP from train any sample within cosine T of any test
     sample. Dedup against the test set, no transitive cascade.
  3. Sweep T. A flat curve means near-dups were not propping the score up.

Usage:  venv-clap/bin/python audit_v2.py
"""
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from memorization_audit import load, N_SPLITS, PROTOTYPES_PER_CLASS

THRESHOLDS = [1.01, 0.99, 0.97, 0.95]  # 1.01 = drop nothing (pack-split only)


def fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    A, B = scaler.transform(Xtr).astype(np.float32), scaler.transform(Xte).astype(np.float32)
    P, lab = [], []
    for c in np.unique(ytr):
        Xc = A[ytr == c]
        k = min(PROTOTYPES_PER_CLASS, len(Xc))
        if k < 1:
            continue
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


def run(X, Xn, y, packs, thresh, shuffle=False):
    yy = y.copy()
    if shuffle:
        np.random.default_rng(0).shuffle(yy)
    y_true, y_pred, dropped, min_train = [], [], 0, 10**9
    for tr, te in GroupKFold(N_SPLITS).split(X, yy, packs):
        keep = np.ones(len(tr), dtype=bool)
        if thresh <= 1.0:
            # drop train rows too close to ANY test row
            A, B = Xn[tr], Xn[te]
            for s in range(0, len(tr), 2048):
                sims = A[s : s + 2048] @ B.T
                keep[s : s + 2048] = sims.max(1) < thresh
        tr2 = tr[keep]
        dropped += len(tr) - len(tr2)
        min_train = min(min_train, len(tr2))
        if len(np.unique(yy[tr2])) < 2:
            continue
        y_true.append(yy[te])
        y_pred.append(fit_predict(X[tr2], yy[tr2], X[te]))
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    return (yt == yp).mean(), dropped, min_train, yt, yp


def main():
    X, y, packs = load()
    Xn = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
    print(f"{len(X)} samples, {len(set(y))} classes, {len(set(packs))} packs")
    sizes = sorted(Counter(packs).values(), reverse=True)
    print(f"pack sizes: largest {sizes[0]} ({sizes[0]/len(X)*100:.1f}% of corpus), median {int(np.median(sizes))}")
    print("-> folds stay balanced because no pack dominates.\n")

    print(f"{'dedup T':>8} {'train dropped':>14} {'min train':>10} {'accuracy':>9}", flush=True)
    last = None
    for t in THRESHOLDS:
        acc, dropped, min_tr, yt, yp = run(X, Xn, y, packs, t)
        tag = "none (pack split only)" if t > 1.0 else f"{t}"
        print(f"{tag:>8} {dropped:>14} {min_tr:>10} {acc*100:>8.2f}%", flush=True)
        last = (t, yt, yp)

    acc0, *_ = run(X, Xn, y, packs, 0.97, shuffle=True)
    chance = max((y == c).mean() for c in set(y))
    print(f"\nnegative control (shuffled labels, T=0.97): {acc0*100:.2f}%  (chance ~ {chance*100:.1f}%)")

    t, yt, yp = last
    print(f"\nper-class accuracy at T={t}:")
    counts = Counter(yt)
    for acc, c, k in sorted((( yp[yt==c]==c).mean(), c, counts[c]) for c in sorted(set(yt))):
        print(f"  {c:<12} n={k:<5} {acc*100:5.1f}%")


if __name__ == "__main__":
    main()
