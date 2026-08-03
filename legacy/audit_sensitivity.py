#!/usr/bin/env python3
"""
Deeper memorization audit: how sensitive is the "leakage-safe" number to the
near-duplicate threshold we picked?

memorization_audit.py holds out same-pack + cosine>=0.98 twins. But 0.98 was a
judgement call. If real accuracy keeps sliding as the threshold tightens, then
near-twins BELOW 0.98 are still leaking and the headline number is optimistic.
If it plateaus, 0.98 was strict enough and the number is trustworthy.

Also reports per-class accuracy at the strictest setting, to show which classes
lean hardest on having seen near-identical audio.

Usage:  venv-clap/bin/python audit_sensitivity.py
"""
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from memorization_audit import UnionFind, load, N_SPLITS, PROTOTYPES_PER_CLASS


def prototype_cv(X, y, groups):
    """Same protocol as memorization_audit.prototype_cv, but distances via
    ||a-b||^2 = |a|^2 + |b|^2 - 2ab instead of materializing the (n_test,
    n_proto, 512) difference tensor — that peaked around 4.7 GB and got the
    process OOM-killed when run repeatedly."""
    y_true, y_pred = [], []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        P, lab = [], []
        for c in np.unique(y[tr]):
            Xc = Xtr[y[tr] == c]
            k = min(PROTOTYPES_PER_CLASS, len(Xc))
            km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(Xc)
            P.append(km.cluster_centers_)
            lab += [c] * k
        P = np.vstack(P).astype(np.float32)
        lab = np.array(lab)
        Xte = Xte.astype(np.float32)
        pn = (P * P).sum(1)
        best = np.empty(len(Xte), dtype=np.int64)
        for s in range(0, len(Xte), 1024):
            blk = Xte[s : s + 1024]
            d2 = (blk * blk).sum(1)[:, None] + pn[None, :] - 2.0 * (blk @ P.T)
            best[s : s + 1024] = d2.argmin(1)
        y_true.append(y[te])
        y_pred.append(lab[best])
    return np.concatenate(y_true), np.concatenate(y_pred)

THRESHOLDS = [0.995, 0.98, 0.95, 0.90]
LOOSEST = min(THRESHOLDS)


def all_pairs(X, min_thresh):
    """Every near-dup pair at the loosest threshold, computed once and reused."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    pairs = []
    chunk = 512
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        sims = Xn[s:e] @ Xn.T
        for local, gi in enumerate(range(s, e)):
            js = np.where(sims[local] >= min_thresh)[0]
            for gj in js:
                if gj > gi:
                    pairs.append((gi, int(gj), float(sims[local, gj])))
    return pairs


def groups_at(n, packs, pairs, thresh):
    uf = UnionFind(n)
    by_pack = {}
    for i, p in enumerate(packs):
        by_pack.setdefault(p, []).append(i)
    for members in by_pack.values():
        for j in members[1:]:
            uf.union(members[0], j)
    for i, j, sim in pairs:
        if sim >= thresh:
            uf.union(i, j)
    return np.array([uf.find(i) for i in range(n)])


def main():
    X, y, packs = load()
    n = len(X)
    print(f"{n} samples, {len(set(y))} classes, {len(set(packs))} packs")
    print(f"computing near-dup pairs once at cosine >= {LOOSEST} ...")
    pairs = all_pairs(X, LOOSEST)
    print(f"  {len(pairs)} candidate pairs\n")

    print(f"{'thresh':>7} {'pairs':>8} {'groups':>7} {'accuracy':>9}", flush=True)
    results = {}
    for t in sorted(THRESHOLDS, reverse=True):
        g = groups_at(n, packs, pairs, t)
        yt, yp = prototype_cv(X, y, g)
        acc = (yt == yp).mean()
        npairs = sum(1 for _, _, s in pairs if s >= t)
        print(f"{t:>7.3f} {npairs:>8} {len(set(g)):>7} {acc*100:>8.2f}%", flush=True)
        results[t] = (acc, yt, yp)

    # Negative control once, at the strictest grouping. Already known clean at
    # 0.98; re-running it per threshold is pure cost.
    strict_g = groups_at(n, packs, pairs, min(THRESHOLDS))
    rng = np.random.default_rng(0)
    ys = y.copy()
    rng.shuffle(ys)
    yt2, yp2 = prototype_cv(X, ys, strict_g)
    chance = max((y == c).mean() for c in set(y))
    print(f"\nnegative control (shuffled labels): {(yt2==yp2).mean()*100:.2f}% "
          f"(chance ~ {chance*100:.1f}%)", flush=True)

    # Per-class at the strictest (most honest) setting.
    strict = min(THRESHOLDS)
    _, yt, yp = results[strict]
    print(f"\nper-class accuracy at the strictest threshold ({strict}):")
    counts = Counter(yt)
    rows = []
    for c in sorted(set(yt)):
        m = yt == c
        rows.append(((yp[m] == c).mean(), c, counts[c]))
    for acc, c, k in sorted(rows):
        print(f"  {c:<12} n={k:<5} {acc*100:5.1f}%")

    spread = (results[max(THRESHOLDS)][0] - results[strict][0]) * 100
    print(f"\naccuracy drop from loosest->strictest dedup: {spread:.2f} pts")
    print("  small (<1pt) => 0.98 was strict enough, headline number holds.")
    print("  large        => near-twins below 0.98 are still inflating it.")


if __name__ == "__main__":
    main()
