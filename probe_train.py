"""
Linear-probe experiment: does a small classifier trained on frozen CLAP
embeddings beat raw zero-shot? And how does accuracy scale with data?

We do NOT train an audio model. We extract CLAP audio embeddings (512-d) once,
cache them, then train a logistic-regression head with cross-validation. Labels
come from the folder names in test_samples/.

This tells us whether collecting more labeled drums will pay off, and roughly
how much we'd need.
"""
import os
import glob
import json
import warnings
import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
MODEL_ID = "laion/clap-htsat-unfused"
SR = 48000
TEST_DIR = "test_samples"
CACHE = "clap_embeddings.npz"
MAX_ONESHOT_SEC = 2.0

FOLDER_TO_CATEGORY = {
    "Snares": "Snare", "Claps": "Clap", "Hats": "Hi-Hat",
    "Open Hats": "Open Hat", "Percs": "Perc", "808s": "Kick/808", "Kicks": "Kick/808",
}

# Zero-shot prompts (same as before) so we can compare probe vs zero-shot head-to-head.
CATEGORIES = {
    "Kick/808": ["a kick drum", "an 808 bass drum", "a deep boomy kick drum hit"],
    "Snare": ["a snare drum", "a snare drum hit", "an acoustic snare"],
    "Clap": ["a hand clap", "a clap sound", "a snappy clap"],
    "Hi-Hat": ["a closed hi-hat", "a short closed hi-hat cymbal tick"],
    "Open Hat": ["an open hi-hat", "a long open hi-hat cymbal"],
    "Perc": ["a percussion hit", "a rimshot", "a shaker", "a tambourine", "a conga"],
}


def gt_for(path):
    return FOLDER_TO_CATEGORY.get(os.path.basename(os.path.dirname(path)))


def build_embeddings():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Extracting CLAP embeddings on {device} (cached to {CACHE})...")
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)

    files = [f for f in glob.glob(f"{TEST_DIR}/**/*", recursive=True)
             if f.lower().endswith((".wav", ".mp3", ".aif", ".aiff"))]
    X, y, names = [], [], []
    for f in files:
        lbl = gt_for(f)
        if lbl is None:
            continue
        try:
            audio, _ = librosa.load(f, sr=SR, mono=True)
            audio, _ = librosa.effects.trim(audio, top_db=30)
        except Exception:
            continue
        if len(audio) == 0 or len(audio) / SR > MAX_ONESHOT_SEC:
            continue
        with torch.no_grad():
            ain = proc(audio=audio, sampling_rate=SR, return_tensors="pt").to(device)
            emb = model.get_audio_features(**ain).pooler_output
            emb = torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy()
        X.append(emb); y.append(lbl); names.append(os.path.basename(f))

    X = np.array(X); y = np.array(y); names = np.array(names)
    np.savez(CACHE, X=X, y=y, names=names)
    return X, y, names


def zero_shot_preds(X):
    """Reproduce the zero-shot head using cached text embeddings for a fair compare."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)
    keys = list(CATEGORIES.keys())
    prompts, owner = [], []
    for k in keys:
        for p in CATEGORIES[k]:
            prompts.append(p); owner.append(k)
    with torch.no_grad():
        tin = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        temb = torch.nn.functional.normalize(model.get_text_features(**tin).pooler_output, dim=-1).cpu().numpy()
    preds = []
    for emb in X:
        sims = emb @ temb.T
        score = {k: [] for k in keys}
        for i, o in enumerate(owner):
            score[o].append(sims[i])
        score = {k: float(np.mean(v)) for k, v in score.items()}
        preds.append(max(score, key=score.get))
    return np.array(preds)


def main():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        X, y, names = d["X"], d["y"], d["names"]
        print(f"Loaded {len(X)} cached embeddings.")
    else:
        X, y, names = build_embeddings()
        print(f"Extracted {len(X)} embeddings.")

    classes = sorted(set(y.tolist()))
    print("Class counts:", {c: int((y == c).sum()) for c in classes})

    # --- Zero-shot baseline on the same items ---
    zs = zero_shot_preds(X)
    zs_acc = (zs == y).mean()

    # --- Linear probe, cross-validated (leave-one-out-ish via 5-fold) ---
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    n_splits = min(5, min(int((y == c).sum()) for c in classes))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probe = cross_val_predict(clf, X, y, cv=cv)
    probe_acc = (probe == y).mean()

    print(f"\n=== Head-to-head on {len(X)} sounds ({n_splits}-fold CV for the probe) ===")
    print(f"  Zero-shot:    {zs_acc*100:5.1f}%")
    print(f"  Linear probe: {probe_acc*100:5.1f}%   <-- trained on frozen CLAP embeddings")

    print("\nPer-class (zero-shot -> probe):")
    for c in classes:
        m = (y == c)
        zc = (zs[m] == y[m]).mean() * 100
        pc = (probe[m] == y[m]).mean() * 100
        print(f"  {c:9s}  n={int(m.sum()):2d}   {zc:5.1f}%  ->  {pc:5.1f}%")

    # --- Learning curve: how does probe accuracy grow with data? ---
    try:
        sizes, train_sc, test_sc = learning_curve(
            clf, X, y, cv=cv, train_sizes=np.linspace(0.3, 1.0, 5),
            scoring="accuracy", random_state=42)
        print("\nLearning curve (CV accuracy vs #training samples):")
        for s, t in zip(sizes, test_sc.mean(axis=1)):
            print(f"  {int(s):3d} samples -> {t*100:5.1f}%")
        print("  (still climbing at the right edge = more data will keep helping)")
    except Exception as e:
        print("learning_curve skipped:", e)


if __name__ == "__main__":
    main()
