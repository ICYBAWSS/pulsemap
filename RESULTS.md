# Classifier accuracy investigation — results & best known approach

Honest, leakage-safe evaluation throughout: GroupKFold over union-find(same
`source_folder` ∨ CLAP-cosine ≥ 0.98), so no pack and no near-duplicate can
straddle train/test. Negative control (shuffled labels → chance) validates the
harness. The number that matters is **balanced accuracy** (the corpus is ~84:1
imbalanced; overall accuracy is dominated by the big classes).

## ✅ BEST KNOWN APPROACH

**Frozen `laion/clap-htsat-unfused` → supervised-contrastive projection → logistic,
with augmented thin classes.** Pipeline:
1. **Preprocessing**: silence-trim at `top_db=60` (not 30 — 30 clips cymbal/ride
   decay tails), decay preserved, CLAP repeatpad.
2. **Label cleanup**: filename-vs-folder relabels (bracket-stripped keyword
   match; 337 fixes, e.g. 808→Bass ×49, Closed→Open Hat ×24, Snare→Clap ×100).
3. **`_unsorted` recovery**: 103 files with confident filenames added as labels.
4. **Supervised-contrastive projection**: small MLP (512→256→128) trained with
   class-weighted CE + supervised-contrastive loss, then logistic on the
   projected space. Learns a better-separated representation ON TOP of frozen
   CLAP (no encoder finetune, no overfit).
5. **Augmentation of thin classes**: light effects (pitch/stretch/gain/noise),
   CLAP-cosine quality gate (keep if ≥0.70 to source AND class unchanged),
   TRAIN-ONLY, each aug tied to its source's leakage group. `augment.py`.

**Result: overall 83.5% / balanced 73.4%, full 20-class taxonomy, leakage-safe.**
Progression (honest balanced): 55.5 (raw) → 62.5 (cleanup) → 65.4 (contrastive)
→ 66.5 (+aug) → 73.4 (+kNN-consensus relabel, 1045 auto-fixes, TRAIN-ONLY so
test stays honest). The dominant lever was fixing ~14.5% mislabeled training
data found via cross-group kNN disagreement. `relabel_prep.py` (find + auto-fix obvious),
`relabel_ui.py` (local ear-review UI for ambiguous, localhost:8777). Human review is now OPTIONAL: a threshold sweep (honest, train-only) showed
auto-fixing more never hurts and plateaus ~72.5 (logistic). Expanded auto-fix
to 1537 (own<0.45 & consensus>=0.6) gives balanced 73.6 / overall 84.5 with
ZERO manual review. The ~2191 weak-consensus leftovers are genuinely ambiguous
(a human squints too); reviewing them buys ~1-2 pts for hours of work. Use the
UI (`relabel_ui.py`) for a 5-min spot-check of auto-fixes, not a full pass.
CAUTION: relabeling ALL data (incl. test) reads 79.7 balanced but is CIRCULAR
(you moved test labels toward the model's opinion). Always relabel TRAIN-ONLY
for the honest number, or use human-verified labels which are real ground truth.
Reproduce: `build_dataset_v2.py` + `augment.py` → embeddings, then
`export_final_model.py` trains the contrastive-projection head. (Contrastive
numbers vary ±~1 run-to-run; no fixed seed.)

SHIPPED (native app): `export_final_model.py` trains the full pipeline on all
cleaned+relabeled+augmented data and writes `native/models/model.json` (20-class,
with projection layers). `classify.rs` runs CLAP → scaler → MLP → L2norm →
logistic; verified bit-for-bit against numpy. Native trim aligned to top_db=60.
Unsorted threshold calibrated to 0.50 (~5% on real sounds). End-to-end checked
on test_samples via the `classify_folder` bin. (The old 18-class linear-head classifier this superseded is gone; see git
history before this cleanup if you need it.)

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

## ✅ What WORKED (stacks, all honest)

| lever | balanced gain | note |
|---|---|---|
| Label cleanup + preprocessing | 55.5 → 62.5 | data quality |
| Supervised-contrastive projection | 62.5 → 65.4 | learned representation on frozen CLAP; the viable form of "train our own model" at our data scale |
| Augmentation of thin classes | 65.4 → 66.5 | Rolls +16, Bass +7 recall; TRAIN-only, gated |

Encoder finetune, encoder swaps, DSP features, raw specialist heads, and the
encoder ensemble did NOT help (see above). The heavy-encoder path is a dead end
at this data scale; the wins are all lightweight (better head/representation +
more/cleaner data).

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
