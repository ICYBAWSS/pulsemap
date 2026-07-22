#!/usr/bin/env python3
"""
Memorization audit: prove the classifier isn't getting credit for sounds it has
effectively already seen. Closes the leak that grouped-by-pack misses —
cross-pack NEAR-DUPLICATES (same sound re-trimmed/renamed in another pack).

Protocol:
  1. Near-duplicate detection in embedding space (cosine >= THRESH).
  2. Leakage groups = union-find over (same source_folder) OR (near-dup pair),
     so neither a pack nor a near-twin can straddle train/test.
  3. GroupKFold on leakage groups -> the *truly* honest accuracy.
  4. Negative control: shuffle labels, re-run. Must collapse to ~chance; if it
     stays high, the harness itself leaks.

Usage:  venv-clap/bin/python memorization_audit.py
"""
import csv
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EMB = "training_data/embeddings_final.npz"
MANIFEST = "training_data/manifest.csv"
THRESH = 0.98          # cosine >= this counts as a near-duplicate
PROTOTYPES_PER_CLASS = 40
N_SPLITS = 5


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def load():
    d = np.load(EMB, allow_pickle=True)
    X, y, paths = np.array(d["X"], dtype=np.float32), np.array(d["y"]), np.array(d["paths"])
    src = {}
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            src[r["hash"][:12]] = r["source_folder"]
    packs = np.array([src.get(os.path.splitext(os.path.basename(str(p)))[0], "UNKNOWN") for p in paths])
    return X, y, packs


def near_dup_groups(X, packs):
    """Union-find over pack membership + near-duplicate pairs. Returns group ids
    and cross-pack near-dup pair count."""
    n = len(X)
    uf = UnionFind(n)

    # 1. union samples sharing a pack
    pack_ids = {}
    for i, p in enumerate(packs):
        pack_ids.setdefault(p, []).append(i)
    for members in pack_ids.values():
        for j in members[1:]:
            uf.union(members[0], j)

    # 2. near-duplicate pairs via chunked cosine on L2-normalized embeddings
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cross_pack_dups = 0
    total_dups = 0
    chunk = 512
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = Xn[s:e] @ Xn.T            # (chunk, n)
        for local, gi in enumerate(range(s, e)):
            js = np.where(sims[local] >= THRESH)[0]
            for gj in js:
                if gj <= gi:
                    continue
                total_dups += 1
                if packs[gi] != packs[gj]:
                    cross_pack_dups += 1
                uf.union(gi, gj)

    groups = np.array([uf.find(i) for i in range(n)])
    return groups, total_dups, cross_pack_dups


def prototype_cv(X, y, groups):
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
        P = np.vstack(P)
        lab = np.array(lab)
        d2 = ((Xte[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        y_true.append(y[te])
        y_pred.append(lab[d2.argmin(1)])
    return np.concatenate(y_true), np.concatenate(y_pred)


def main():
    X, y, packs = load()
    print(f"{len(X)} samples, {len(set(y))} classes, {len(set(packs))} packs")

    groups, total_dups, cross_dups = near_dup_groups(X, packs)
    print(f"\nnear-duplicate pairs (cosine >= {THRESH}): {total_dups}")
    print(f"  of which CROSS-PACK (the hidden leak): {cross_dups}")
    print(f"leakage groups after merging packs + near-dups: {len(set(groups))} "
          f"(was {len(set(packs))} packs)")

    yt, yp = prototype_cv(X, y, groups)
    acc = (yt == yp).mean()
    print(f"\n>>> LEAKAGE-SAFE accuracy (packs + near-dups held out): {acc*100:.2f}%")
    print("    (compare: 82.3% random, 76.4% grouped-by-pack)")

    # Negative control: shuffled labels must collapse to ~chance.
    rng = np.random.default_rng(0)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    yt2, yp2 = prototype_cv(X, y_shuf, groups)
    chance = max((y == c).mean() for c in set(y))
    print(f"\nnegative control (shuffled labels): {(yt2==yp2).mean()*100:.2f}%  "
          f"(chance ~ {chance*100:.1f}% = largest class prior)")
    print("  -> if this is near chance, the harness has no label leakage.")


if __name__ == "__main__":
    main()
