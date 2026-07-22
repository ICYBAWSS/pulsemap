"""Reference embedding for the Rust port to verify against.

Dumps, for one test clip:
  - input_features.npy  : the ClapProcessor output fed to the audio model
  - is_longer.npy       : the companion bool flag
  - embedding_ref.npy   : ClapAudioModelWithProjection audio_embeds (raw)
  - embedding_ref_norm.npy : L2-normalized (what build_map.py actually uses)
  - audio_48k_mono.npy  : the decoded waveform, so Rust can verify its own
                          mel preprocessing against input_features independently.

Run: venv-clap/bin/python native/ref/clap_ref.py [wav]
"""
import sys
import numpy as np
import librosa
import torch
from transformers import ClapAudioModelWithProjection, ClapProcessor

MODEL_ID = "laion/clap-htsat-unfused"
SR = 48000
OUT = __file__.rsplit("/", 1)[0]

wav = sys.argv[1] if len(sys.argv) > 1 else (
    "test_samples/BD AXIOM (DEMO)/AXIOM (DEMO)/KICK/BD 808ISH² KICK.wav"
)

# 1. Decode -> 48k mono (mirrors build_map.py's librosa load).
audio, _ = librosa.load(wav, sr=SR, mono=True)
print(f"audio: {audio.shape} samples ({audio.shape[0]/SR:.3f}s)")
np.save(f"{OUT}/audio_48k_mono.npy", audio.astype(np.float32))

# 2. Processor -> input_features (+ is_longer).
proc = ClapProcessor.from_pretrained(MODEL_ID)
inp = proc(audio=audio, sampling_rate=SR, return_tensors="pt")
for k, v in inp.items():
    print(f"proc[{k}]: shape={tuple(v.shape)} dtype={v.dtype}")
np.save(f"{OUT}/input_features.npy", inp["input_features"].numpy().astype(np.float32))
if "is_longer" in inp:
    np.save(f"{OUT}/is_longer.npy", inp["is_longer"].numpy())

# 3. Model -> audio embedding (matches onnx/audio_model.onnx).
model = ClapAudioModelWithProjection.from_pretrained(MODEL_ID).eval()
with torch.no_grad():
    out = model(**inp)
emb = out.audio_embeds.numpy().astype(np.float32)  # (1, 512)
print(f"embedding: shape={emb.shape} | first5={emb[0, :5]}")
np.save(f"{OUT}/embedding_ref.npy", emb)
emb_norm = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
np.save(f"{OUT}/embedding_ref_norm.npy", emb_norm)
print("wrote reference tensors to", OUT)
