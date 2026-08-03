"""
Probe: can zero-shot CLAP name SPECIFIC percussion types, instead of a vague
'Perc/Other' bin? We give it a rich, granular vocabulary and inspect what the
real percussion samples get called (top-3 each). This directly tests the worry
that the vague/unlabeled percussion sounds -- the ones humans never sort -- can
still be handled.
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
SR = 48000
TEST_DIR = "test_samples"

# A RICH vocabulary -- granular, not coarse. Zero-shot makes adding types free.
VOCAB = {
    "Kick/808":   ["a kick drum", "an 808 bass drum"],
    "Snare":      ["a snare drum", "an acoustic snare hit"],
    "Clap":       ["a hand clap", "a snappy clap"],
    "Snap":       ["a finger snap"],
    "Rimshot":    ["a rimshot", "a rim click on a snare drum"],
    "Closed Hat": ["a closed hi-hat", "a short closed hi-hat tick"],
    "Open Hat":   ["an open hi-hat", "a long open hi-hat cymbal"],
    "Cymbal":     ["a crash cymbal", "a splash cymbal"],
    "Ride":       ["a ride cymbal"],
    "Tom":        ["a tom drum", "a floor tom hit"],
    "Shaker":     ["a shaker", "a shaker percussion"],
    "Tambourine": ["a tambourine"],
    "Conga":      ["a conga drum", "a bongo drum"],
    "Cowbell":    ["a cowbell"],
    "Woodblock":  ["a woodblock", "a click percussion"],
    "Vocal":      ["a vocal sample", "a human voice chant"],
}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading CLAP ({MODEL_ID}) on {device}...  |  {len(VOCAB)} categories\n")
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    proc = ClapProcessor.from_pretrained(MODEL_ID)

    keys = list(VOCAB.keys())
    prompts, owner = [], []
    for k in keys:
        for p in VOCAB[k]:
            prompts.append(p); owner.append(k)
    with torch.no_grad():
        tin = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        temb = torch.nn.functional.normalize(model.get_text_features(**tin).pooler_output, dim=-1)

    def classify(y):
        with torch.no_grad():
            ain = proc(audio=y, sampling_rate=SR, return_tensors="pt").to(device)
            aemb = torch.nn.functional.normalize(model.get_audio_features(**ain).pooler_output, dim=-1)
            sims = (aemb @ temb.T).squeeze(0).cpu().numpy()
        score = {k: [] for k in keys}
        for i, o in enumerate(owner):
            score[o].append(sims[i])
        score = {k: float(np.mean(v)) for k, v in score.items()}
        return sorted(score.items(), key=lambda kv: -kv[1])

    def load(f):
        y, _ = librosa.load(f, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)
        return y

    # --- The real test: the messy Perc folder. What does each get named? ---
    perc_files = sorted(glob.glob(f"{TEST_DIR}/**/Percs/*", recursive=True))
    perc_files = [f for f in perc_files if f.lower().endswith((".wav", ".mp3", ".aif", ".aiff"))]
    print("=== PERCUSSION probe (filename hints the true type) ===")
    for f in perc_files:
        try:
            top = classify(load(f))[:3]
        except Exception:
            continue
        top_str = "   ".join(f"{k}:{s:.2f}" for k, s in top)
        print(f"  {os.path.basename(f):45s} -> {top_str}")

    # --- Sanity: do the strong classes survive the richer vocabulary? ---
    print("\n=== Sanity check on clearly-labeled folders (top-1) ===")
    for folder in ["Snares", "Hats", "808s", "Claps"]:
        fs = [f for f in glob.glob(f"{TEST_DIR}/**/{folder}/*", recursive=True)
              if f.lower().endswith((".wav", ".mp3", ".aif", ".aiff"))]
        preds = []
        for f in fs:
            try:
                preds.append(classify(load(f))[0][0])
            except Exception:
                pass
        from collections import Counter
        dist = ", ".join(f"{k}:{c}" for k, c in Counter(preds).most_common())
        print(f"  {folder:8s} ({len(preds)}) -> {dist}")


if __name__ == "__main__":
    main()
