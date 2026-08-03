#!/usr/bin/env python3
"""
Find and prune likely-mislabeled training samples (leaked kits have plenty).

Method: out-of-fold predicted probabilities. If the model -- trained on OTHER
folds -- is very confident a sample is class B while its folder says class A,
the folder label is probably wrong. We flag those, spot-check against the
original filenames (from manifest.csv), prune the most confident cases, and
re-measure.

Honest caveat: pruning by model disagreement removes true noise AND some hard-
but-correct cases, so the post-prune CV number is a mild upper bound. The point
is a cleaner training set for the deployed model, plus a flagged list to eyeball.

Usage:  source venv-clap/bin/activate && python label_clean.py
"""
import csv
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

EMB = "training_data/embeddings.npz"
MANIFEST = "training_data/manifest.csv"
FLAGGED_OUT = "training_data/flagged_labels.csv"
PRUNE_THRESH = 0.02   # given-label prob below this AND another class dominant -> prune


def load_orig_names():
    """hash(prefix) -> (orig_name, source_folder) from the manifest, for spot-checks."""
    m = {}
    try:
        with open(MANIFEST) as f:
            for row in csv.DictReader(f):
                m[row["hash"][:12]] = (row.get("orig_name", ""), row.get("source_folder", ""))
    except FileNotFoundError:
        pass
    return m


def svc_cv(X, y, labels):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), SVC(C=8, gamma="scale", kernel="rbf",
                                              class_weight="balanced", cache_size=1500))
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    acc = (pred == y).mean()
    bal = np.mean([(pred[y == c] == c).mean() for c in labels])
    return acc, bal


def main():
    d = np.load(EMB, allow_pickle=True)
    X, y, paths = np.array(d["X"]), np.array(d["y"]), np.array(d["paths"])
    labels = sorted(set(y))
    idx = {c: i for i, c in enumerate(labels)}
    counts = Counter(y)
    names = load_orig_names()

    print(f"{len(X)} samples, {len(labels)} classes.\n")
    print("Baseline (SVC, all data):")
    a0, b0 = svc_cv(X, y, labels)
    print(f"  overall {a0*100:.1f}%   balanced {b0*100:.1f}%\n")

    # Out-of-fold probabilities from a fast calibrated-ish model.
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced")),
        X, y, cv=cv, method="predict_proba", n_jobs=-1)

    p_given = np.array([proba[i, idx[y[i]]] for i in range(len(y))])
    pred_lbl = np.array([labels[j] for j in proba.argmax(1)])
    p_pred = proba.max(1)

    flag = (p_given < PRUNE_THRESH) & (pred_lbl != y)
    print(f"Flagged {flag.sum()} / {len(y)} ({100*flag.sum()/len(y):.1f}%) as likely-mislabeled "
          f"(given-label prob < {PRUNE_THRESH}).")

    # Per-class flag rate.
    print("\nFlag rate by class (how noisy each folder-label looks):")
    per = defaultdict(lambda: [0, 0])
    for i in range(len(y)):
        per[y[i]][1] += 1
        if flag[i]:
            per[y[i]][0] += 1
    for c in sorted(per, key=lambda k: -per[k][0] / per[k][1]):
        f, n = per[c]
        print(f"  {c:12s} {f:4d}/{n:<4d} ({100*f/n:4.1f}%)")

    # Spot-check: show 15 flagged with their original filenames as corroboration.
    print("\nSpot-check (folder-label -> model says  |  original filename):")
    shown = 0
    order = np.argsort(p_given)
    with open(FLAGGED_OUT, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["given_label", "model_label", "p_given", "p_model", "orig_name", "source_folder"])
        for i in order:
            if not flag[i]:
                continue
            h = paths[i].split("/")[-1].split(".")[0]
            orig, folder = names.get(h, ("", ""))
            w.writerow([y[i], pred_lbl[i], f"{p_given[i]:.3f}", f"{p_pred[i]:.3f}", orig, folder])
            if shown < 15:
                print(f"  {y[i]:11s} -> {pred_lbl[i]:11s}  |  {orig[:48]}")
                shown += 1
    print(f"\n(full list -> {FLAGGED_OUT})")

    # Prune and re-measure.
    keep = ~flag
    Xc, yc = X[keep], y[keep]
    print(f"\nAfter pruning {flag.sum()} samples (SVC on cleaned set):")
    a1, b1 = svc_cv(Xc, yc, labels)
    print(f"  overall {a1*100:.1f}% ({a1*100-a0*100:+.1f})   balanced {b1*100:.1f}% ({b1*100-b0*100:+.1f})")
    print("  (mild upper bound — removes some hard-but-correct cases too)")


if __name__ == "__main__":
    main()
