#!/usr/bin/env python3
"""
Sweep classifier heads on cached CLAP embeddings. Embeddings are frozen, so this
is fast -- we're only searching for the best head + hyperparameters.

Reports 5-fold CV overall accuracy AND balanced (mean per-class) accuracy, since
the classes are very imbalanced. Balanced is the number we optimize.

Usage:  source venv-clap/bin/activate && python experiments.py [emb.npz]
"""
import sys
import time
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

CACHE = sys.argv[1] if len(sys.argv) > 1 else "training_data/embeddings.npz"


def evaluate(name, clf, X, y, cv, labels, counts):
    t0 = time.time()
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    acc = (pred == y).mean()
    bal = np.mean([(pred[y == c] == c).mean() for c in labels])
    print(f"  {name:26s} overall {acc*100:5.1f}%   balanced {bal*100:5.1f}%   "
          f"({time.time()-t0:.0f}s)")
    return name, acc, bal, pred


def main():
    d = np.load(CACHE, allow_pickle=True)
    X, y = np.array(d["X"]), np.array(d["y"])
    counts = Counter(y)
    labels = sorted(counts.keys())
    print(f"{CACHE}: {len(X)} embeddings, {len(labels)} classes\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sc = StandardScaler

    candidates = [
        ("logreg C=1 (baseline)", make_pipeline(sc(), LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced"))),
        ("logreg C=5", make_pipeline(sc(), LogisticRegression(
            max_iter=3000, C=5.0, class_weight="balanced"))),
        ("logreg C=10", make_pipeline(sc(), LogisticRegression(
            max_iter=3000, C=10.0, class_weight="balanced"))),
        ("linsvc C=1", make_pipeline(sc(), LinearSVC(C=1.0, class_weight="balanced"))),
        ("knn k=15 cosine", make_pipeline(sc(), KNeighborsClassifier(
            n_neighbors=15, weights="distance", metric="cosine"))),
        ("mlp (256,)", make_pipeline(sc(), MLPClassifier(
            hidden_layer_sizes=(256,), max_iter=300, early_stopping=True,
            random_state=42))),
        ("mlp (512,256)", make_pipeline(sc(), MLPClassifier(
            hidden_layer_sizes=(512, 256), max_iter=300, early_stopping=True,
            random_state=42))),
        ("hist-gbm", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=42)),
        ("svc rbf C=8", make_pipeline(sc(), SVC(
            C=8.0, kernel="rbf", class_weight="balanced", cache_size=1000))),
    ]

    print("Sweeping heads (5-fold CV):")
    results = []
    for name, clf in candidates:
        try:
            results.append(evaluate(name, clf, X, y, cv, labels, counts))
        except Exception as e:
            print(f"  {name:26s} FAILED: {e}")

    results.sort(key=lambda r: r[2], reverse=True)
    print(f"\nBest: {results[0][0]}  (balanced {results[0][2]*100:.1f}%)")


if __name__ == "__main__":
    main()
