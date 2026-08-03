"""
Zero-shot CLAP classification test for PulseMap.

Goal: prove that a pretrained audio+text model (CLAP) can file each sound into a
named, human-recognizable category ("Snare", "Kick", "Hi-Hat"...) WITHOUT any
training or user labels -- and that it does so consistently.

We use the neatly-foldered test_samples/ as an ANSWER KEY (ground truth) to score
the predictions. In the real product no labels exist; the folders are only here so
we can measure accuracy.
"""
import os
import glob
import warnings
import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

MODEL_ID = os.environ.get("CLAP_MODEL", "laion/clap-htsat-unfused")
SAMPLE_RATE = 48000  # CLAP expects 48kHz
TEST_DIR = "test_samples"

# Only classify one-shots here; loops go down a different path in the real product.
MAX_ONESHOT_SEC = 2.0

# --- The vocabulary: the "sections" of the canvas. ---------------------------
# Each category maps to one or more natural-language prompts. We give CLAP several
# phrasings per class and take the best-matching one -- prompt wording matters a lot.
CATEGORIES = {
    "Kick/808":   ["a kick drum", "an 808 bass drum", "a deep boomy kick drum hit"],
    "Snare":      ["a snare drum", "a snare drum hit", "an acoustic snare"],
    "Clap":       ["a hand clap", "a clap sound", "a snappy clap"],
    "Hi-Hat":     ["a closed hi-hat", "a short closed hi-hat cymbal tick"],
    "Open Hat":   ["an open hi-hat", "a long open hi-hat cymbal"],
    "Perc":       ["a percussion hit", "a rimshot", "a shaker", "a tambourine", "a conga"],
}

# Map a file's ground-truth folder to one of our category keys.
FOLDER_TO_CATEGORY = {
    "Snares": "Snare",
    "Claps": "Clap",
    "Hats": "Hi-Hat",
    "Open Hats": "Open Hat",
    "Percs": "Perc",
    "808s": "Kick/808",
    "Kicks": "Kick/808",
}


def ground_truth_for(path):
    """Derive the true label from the parent folder name, if we recognize it."""
    parent = os.path.basename(os.path.dirname(path))
    return FOLDER_TO_CATEGORY.get(parent)


def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading CLAP ({MODEL_ID}) on {device}...")
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)

    # Precompute a text embedding for every prompt.
    cat_keys = list(CATEGORIES.keys())
    all_prompts, prompt_owner = [], []
    for k in cat_keys:
        for p in CATEGORIES[k]:
            all_prompts.append(p)
            prompt_owner.append(k)
    with torch.no_grad():
        tin = processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
        temb = model.get_text_features(**tin).pooler_output  # transformers 5.x returns an output obj
        temb = torch.nn.functional.normalize(temb, dim=-1)

    # Gather one-shot files that have a known ground-truth folder.
    files = [f for f in glob.glob(f"{TEST_DIR}/**/*", recursive=True)
             if f.lower().endswith((".wav", ".mp3", ".aif", ".aiff"))]

    rows = []
    for f in files:
        gt = ground_truth_for(f)
        if gt is None:
            continue
        try:
            y = load_audio(f)
        except Exception:
            continue
        dur = len(y) / SAMPLE_RATE
        if dur == 0 or dur > MAX_ONESHOT_SEC:
            continue

        with torch.no_grad():
            ain = processor(audio=y, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
            aemb = model.get_audio_features(**ain).pooler_output
            aemb = torch.nn.functional.normalize(aemb, dim=-1)
            sims = (aemb @ temb.T).squeeze(0).cpu().numpy()  # cosine sim to each prompt

        # Average similarity per category (avoids bias toward categories with more
        # prompts), then argmax over categories.
        cat_sims = {k: [] for k in cat_keys}
        for i, owner in enumerate(prompt_owner):
            cat_sims[owner].append(sims[i])
        cat_score = {k: float(np.mean(v)) for k, v in cat_sims.items()}
        pred = max(cat_score, key=cat_score.get)
        conf = cat_score[pred]
        rows.append((os.path.basename(f), gt, pred, conf))

    # --- Report -------------------------------------------------------------
    if not rows:
        print("No labeled one-shots found. Check test_samples/ layout.")
        return

    correct = sum(1 for _, gt, pred, _ in rows if gt == pred)
    print(f"\n=== FINE taxonomy: {correct}/{len(rows)} = {100*correct/len(rows):.1f}% ===\n")

    # Per-class accuracy
    print("Per-class accuracy:")
    for k in cat_keys:
        cls = [r for r in rows if r[1] == k]
        if cls:
            c = sum(1 for r in cls if r[2] == k)
            print(f"  {k:10s} {c:2d}/{len(cls):2d}  ({100*c/len(cls):5.1f}%)")

    # --- COARSE taxonomy: merge the instrument families that share a "section" ---
    # These are the canvas SECTIONS; UMAP separates the fine detail *within* each.
    COARSE = {
        "Kick/808": "Kick",
        "Snare": "Snare/Clap", "Clap": "Snare/Clap",
        "Hi-Hat": "Hats", "Open Hat": "Hats",
        "Perc": "Perc",
    }
    coarse_correct = sum(1 for _, gt, pred, _ in rows if COARSE[gt] == COARSE[pred])
    print(f"\n=== COARSE taxonomy (Kick / Snare+Clap / Hats / Perc): "
          f"{coarse_correct}/{len(rows)} = {100*coarse_correct/len(rows):.1f}% ===")

    # --- Confidence floor: low-confidence sounds go to an 'Unsorted' review pile ---
    for floor in (0.35, 0.40, 0.45):
        confident = [r for r in rows if r[3] >= floor]
        if confident:
            c_acc = 100 * sum(1 for _, gt, pred, _ in confident if COARSE[gt] == COARSE[pred]) / len(confident)
            print(f"  floor={floor:.2f}: {len(confident):2d}/{len(rows)} auto-sorted, "
                  f"{100*len(confident)/len(rows):4.0f}% coverage, {c_acc:5.1f}% coarse-accurate")

    # Confusion (true -> predicted)
    print("\nFINE misclassifications (true -> predicted  [conf]  file):")
    for fn, gt, pred, conf in sorted(rows, key=lambda r: r[1]):
        if gt != pred:
            flag = "" if COARSE[gt] == COARSE[pred] else "  <-- crosses sections"
            print(f"  {gt:9s} -> {pred:9s}  [{conf:.3f}]  {fn}{flag}")


if __name__ == "__main__":
    main()
