# Classifier accuracy investigation — results & best known approach

Honest, leakage-safe evaluation throughout: GroupKFold over union-find(same
`source_folder` ∨ CLAP-cosine ≥ 0.98), so no pack and no near-duplicate can
straddle train/test. Negative control (shuffled labels → chance) validates the
harness. The number that matters is **balanced accuracy** (the corpus is ~84:1
imbalanced; overall accuracy is dominated by the big classes).

## ✅ BEST KNOWN APPROACH (ship this)

**Frozen `laion/clap-htsat-unfused` embeddings + logistic head**, with:
1. **Preprocessing**: silence-trim at `top_db=60` (not 30 — 30 clips cymbal/ride
   decay tails), decay preserved, CLAP repeatpad.
2. **Label cleanup**: filename-vs-folder relabels (bracket-stripped keyword
   match; 337 fixes, e.g. 808→Bass ×49, Closed→Open Hat ×24, Snare→Clap ×100).
3. **`_unsorted` recovery**: 103 files with confident filenames added as labels.

**Result: overall 74.7% / balanced 62.5%, full 20-class taxonomy, leakage-safe.**
Up from balanced 55.5% before cleanup. Reproduce: `build_dataset_v2.py` →
`embeddings_v2.npz`, then a class-weighted `LogisticRegression(C=0.3)`.

Taxonomy is a product requirement — do NOT merge classes (open/closed hat,
cymbal/crash/ride, bass/808 are distinct to producers). Fix representation or
data, never collapse a class.

## ❌ What was tried and did NOT beat the baseline

| lever | honest result | note |
|---|---|---|
| Full CLAP finetune (M5, overnight) | ~59–61% balanced | overfits: huge embedding drift in 2 epochs, best-val@ep2. Verified it trained (cosine 0.52 vs frozen), just doesn't generalize. |
| Encoder swap → `larger_clap_music` | 33% (metallic) | tuned for full mixes, worse on one-shots |
| Encoder swap → `larger_clap_general` | 46% (metallic) | ≈ current, no win |
| DSP feature augmentation | no gain | decay/centroid/flatness concat; CLAP dominates |
| Within-family specialist heads | no gain | metallic info isn't in frozen CLAP embeddings |

**Pattern: every model-side lever failed. The only wins were data-quality
(label cleanup, preprocessing).** The lever is data, not model.

## Where the errors are

- **Solved** (~85% of data): Kick 92, 808 88, Snare 79, Vocal 87, Closed Hat 78, Clap 72.
- **`Cymbal` is a mislabeled grab-bag**: 3% kNN purity — samples labeled "Cymbal"
  almost never resemble each other. Crash (48%) and Ride (40%) ARE coherent.
  Fix = re-curate Cymbal, not merge.
- **Data-starved**: Bass 120, Ride 143, Rolls 103, Woodblock 53 — too few, and
  single-pack classes read 0% under GroupKFold until a 2nd source pack exists.

## Next levers (untested / in progress)

1. **AudioSet-pretrained encoders** (AST / PANNs / BEATs) — percussion-aware.
2. **Encoder ensemble** — clap-general nails Open Hat (54 vs 26) where unfused fails.
3. **Rigorous augmentation of thin classes** — TRAIN-ONLY, each aug tied to its
   source's leakage group, TEST stays 100% real. Multiplies examples; adds no
   new information (won't fix crash-vs-ride).
4. **Pack-diverse collection** for the still-climbing classes: Crash, Open Hat, Clap.

## Reality check on the target

90% *balanced* across 20 fine classes with no leakage is likely beyond the
task's ceiling (crash-vs-ride from a bare one-shot is ambiguous to humans too).
90% *overall* is plausible — the common classes are already there.
