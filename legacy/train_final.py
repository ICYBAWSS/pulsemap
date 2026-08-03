#!/usr/bin/env python3
"""
Production trainer: the winning recipe from the tuning loop.

  embeddings (vanilla CLAP)  ->  prune label noise  ->  SVC-RBF (C=8)

Saves the fitted model + scaler + label list to training_data/probe.joblib and
the pruned-hash list, then reports honest cross-validated per-class recall on the
cleaned set.

Usage:  source venv-clap/bin/activate && python train_final.py
"""
import json
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import joblib

EMB = "training_data/embeddings.npz"
MODEL_OUT = "training_data/probe.joblib"
PRUNED_OUT = "training_data/pruned_hashes.json"
PRUNE_THRESH = 0.02


def flag_noise(X, y, labels):
    """Out-of-fold: flag samples whose given label the model is very sure is wrong."""
    idx = {c: i for i, c in enumerate(labels)}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced")),
        X, y, cv=cv, method="predict_proba", n_jobs=-1)
    p_given = np.array([proba[i, idx[y[i]]] for i in range(len(y))])
    pred = np.array([labels[j] for j in proba.argmax(1)])
    return (p_given < PRUNE_THRESH) & (pred != y)


def main():
    d = np.load(EMB, allow_pickle=True)
    X, y, paths = np.array(d["X"]), np.array(d["y"]), np.array(d["paths"])
    labels = sorted(set(y))

    flag = flag_noise(X, y, labels)
    print(f"Pruning {flag.sum()} / {len(y)} likely-mislabeled samples.")
    pruned_hashes = [paths[i].split("/")[-1].split(".")[0] for i in range(len(y)) if flag[i]]
    json.dump(pruned_hashes, open(PRUNED_OUT, "w"))

    keep = ~flag
    Xc, yc = X[keep], y[keep]
    counts = Counter(yc)

    # Honest CV on the cleaned set.
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), SVC(C=8, gamma="scale", kernel="rbf",
                                              class_weight="balanced", cache_size=1500))
    pred = cross_val_predict(clf, Xc, yc, cv=cv, n_jobs=-1)
    acc = (pred == yc).mean()
    rec = {c: (pred[yc == c] == c).mean() for c in labels}
    bal = np.mean(list(rec.values()))

    print(f"\n=== Cleaned CV: overall {acc*100:.1f}%   balanced {bal*100:.1f}% ===\n")
    print("Per-class recall (n after cleaning):")
    for c in sorted(labels, key=lambda k: rec[k]):
        print(f"  {c:12s} {rec[c]*100:5.1f}%  (n={counts[c]})")

    cm = confusion_matrix(yc, pred, labels=labels)
    conf = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels))
            for j in range(len(labels)) if i != j and cm[i, j] > 0]
    print("\nTop remaining confusions (true -> predicted, count):")
    for cnt, tc, pc in sorted(conf, reverse=True)[:10]:
        print(f"  {tc:12s} -> {pc:12s} {cnt}")

    # Fit and save the production model on all cleaned data.
    clf.fit(Xc, yc)
    joblib.dump({"model": clf, "labels": labels, "prune_thresh": PRUNE_THRESH}, MODEL_OUT)
    print(f"\nSaved production model -> {MODEL_OUT}  ({len(Xc)} clean training samples)")


if __name__ == "__main__":
    main()
