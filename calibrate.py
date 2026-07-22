#!/usr/bin/env python3
"""
Why does the Unsorted pile contain the wrong things?

The shipped rule is `margin = 2nd_best_class_dist - best_class_dist >= 0.20`.
That only asks WHICH class won, never whether the sound resembles ANY class.
A sound unlike anything in training sits far from every prototype, but if one
class happens to edge out the next it still gets filed confidently.

This measures, on honest pack-split CV, how well two signals predict whether a
classification is actually correct:
  - best_dist : distance to the nearest prototype  (an out-of-distribution cue)
  - margin    : gap to the runner-up class         (the current rule)

Usage:  venv-clap/bin/python calibrate.py
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from memorization_audit import load, N_SPLITS, PROTOTYPES_PER_CLASS


def collect():
    X, y, packs = load()
    rows = []
    for tr, te in GroupKFold(N_SPLITS).split(X, y, packs):
        scaler = StandardScaler().fit(X[tr])
        A, B = scaler.transform(X[tr]).astype(np.float32), scaler.transform(X[te]).astype(np.float32)
        classes = np.unique(y[tr])
        P, lab = [], []
        for c in classes:
            Xc = A[y[tr] == c]
            k = min(PROTOTYPES_PER_CLASS, len(Xc))
            km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(Xc)
            P.append(km.cluster_centers_)
            lab += [c] * k
        P = np.vstack(P).astype(np.float32)
        lab = np.array(lab)
        pn = (P * P).sum(1)
        for s in range(0, len(B), 1024):
            blk = B[s : s + 1024]
            d2 = (blk * blk).sum(1)[:, None] + pn[None, :] - 2.0 * (blk @ P.T)
            d = np.sqrt(np.maximum(d2, 0))
            # nearest prototype distance per class
            per = np.stack([d[:, lab == c].min(1) for c in classes], 1)
            order = np.argsort(per, 1)
            best = per[np.arange(len(blk)), order[:, 0]]
            second = per[np.arange(len(blk)), order[:, 1]]
            pred = classes[order[:, 0]]
            truth = y[te][s : s + 1024]
            for bd, sd, p, t in zip(best, second, pred, truth):
                rows.append((float(bd), float(sd - bd), p == t))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def auc(score, correct):
    """P(score of a correct call < score of an incorrect call). 0.5 = useless."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = correct.sum(), (~correct).sum()
    return (r[~correct].sum() - n0 * (n0 + 1) / 2) / (n1 * n0)


def main():
    best, margin, correct = collect()
    print(f"{len(best)} predictions, {correct.mean()*100:.1f}% correct\n")

    print("How well does each signal flag a WRONG classification? (0.5 = useless)")
    print(f"  best_dist (distance to nearest prototype): AUC {auc(best, correct):.3f}")
    print(f"  margin    (gap to runner-up class)       : AUC {auc(-margin, correct):.3f}")

    print("\nbest_dist percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(best, p):.2f}")

    print("\nAccuracy by best_dist decile (is 'far from everything' really worse?):")
    edges = np.percentile(best, np.arange(0, 101, 10))
    for i in range(10):
        m = (best >= edges[i]) & (best <= edges[i + 1])
        if m.sum():
            print(f"  d {edges[i]:6.2f}-{edges[i+1]:6.2f}  n={m.sum():5d}  acc {correct[m].mean()*100:5.1f}%")

    print("\nCurrent rule (margin >= 0.20):")
    keep = margin >= 0.20
    print(f"  classified {keep.mean()*100:.1f}% of sounds, accuracy on them {correct[keep].mean()*100:.1f}%")
    print(f"  sent to Unsorted {(~keep).mean()*100:.1f}%, of which {correct[~keep].mean()*100:.1f}% would have been RIGHT")

    print("\nAlternative: reject when far from everything (best_dist > cutoff).")
    print(f"{'cutoff':>8} {'classified':>11} {'acc on those':>13} {'correct lost':>13}")
    for p in [99, 97, 95, 90, 85, 80]:
        cut = np.percentile(best, p)
        k = best <= cut
        print(f"{cut:>8.2f} {k.mean()*100:>10.1f}% {correct[k].mean()*100:>12.1f}% {correct[~k].mean()*100:>12.1f}%")


if __name__ == "__main__":
    main()


def sweep():
    """Find a better operating point: combine both signals and map the frontier."""
    best, margin, correct = collect()
    rel = margin / np.maximum(best, 1e-6)  # scale-free margin
    print(f"\n  relative margin (margin/best_dist)      : AUC {auc(-rel, correct):.3f}")

    print("\nFrontier — reject if margin < M or best_dist > D:")
    print(f"{'M':>6} {'D':>7} {'classified':>11} {'acc on those':>13} {'wrong filed':>12}")
    rows = []
    for m in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        for dp in [100, 95, 90, 85]:
            d = np.percentile(best, dp)
            keep = (margin >= m) & (best <= d)
            if keep.sum() < 100:
                continue
            cov, acc = keep.mean(), correct[keep].mean()
            # wrong-but-filed as a share of ALL sounds: the thing the user sees
            wrong = (keep & ~correct).mean()
            rows.append((m, d, cov, acc, wrong))
            print(f"{m:>6.1f} {d:>7.1f} {cov*100:>10.1f}% {acc*100:>12.1f}% {wrong*100:>11.1f}%")
    return rows


if __name__ == "__main__" and len(__import__("sys").argv) > 1:
    sweep()
