# PulseMap

A desktop app that maps a folder of drum samples into a visual space you can
browse by ear. Every sound is embedded with a CLAP audio model. It organizes each sound into
a section (Kick, Snare, Closed Hat, …) and laid out within that sections that similar sounds end up together.

Everything runs locally.

## What it does

- **Hover to audition.** Move over a node and it plays, with its waveform in the
  corner.
- **Drag into your DAW.** A plain drag hands the actual file to the OS, so you
  can drop it straight onto a track.
- **Cmd/Ctrl-drag to reclassify.** Pull a sound onto another group and it is
  re-placed next to its nearest match inside that group, gradient and all.
  Corrections persist to `~/.pulsemap/corrections.json` and survive rebuilds.
- **Add folders incrementally.** The map re-solves over the union; already
  analyzed files hit the embedding cache and skip the model entirely.
- **Search** filters by filename or section as you type.

## Running from source

Needs a Rust toolchain. The CLAP encoder weights are not in git (117 MB) —
download `audio_model.onnx` from
[Hugging Face](https://huggingface.co/icybawss/clap-htsat-unfused-audio-encoder-onnx)
and put it in `native/models/`:

```sh
curl -L -o native/models/audio_model.onnx \
  https://huggingface.co/icybawss/clap-htsat-unfused-audio-encoder-onnx/resolve/main/audio_model.onnx
```

```sh
cd native
cargo run --release
```

`native/models/` should then contain:

```
audio_model.onnx     # from Hugging Face (above)
model.json           # classifier head, in git
mel_slaney.npy       # mel filterbank, in git
```


## Layout

| Path | What |
| --- | --- |
| `native/` | The app: Rust, wgpu, winit |
| `native/src/layout.rs` | Section packing and the within-section neighbour embedding |
| `native/src/analyzer.rs` | Decode → mel → CLAP embedding → classify |
| `native/src/main.rs` | Render loop, HUD, interaction |
| `*.py` | Training and evaluation scripts for the classifier head |

## Accuracy

**~85% overall / ~75% balanced accuracy** across the full 20-class taxonomy,
measured on a leakage-safe held-out split (no sample pack or near-duplicate
crosses train/test). Balanced accuracy weights every class equally, so it's
dragged down by rarer, genuinely ambiguous classes (rolls vs. sustained snares,
closed vs. open hats at short decays); overall accuracy reflects what you'd
actually feel using it. Full investigation and numbers in [RESULTS.md](RESULTS.md).

Reclassifying by drag is the intended escape hatch, and corrections feed back
into the layout.
