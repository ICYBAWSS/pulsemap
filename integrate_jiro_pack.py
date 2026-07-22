#!/usr/bin/env python3
"""
Fold the "Jiro Inagaki & Soul Media - Funky Stuff Pack" into the labeled
training corpus, following collect.py's exact convention: md5-hash the file,
copy to training_data/labeled/<Label>/<hash[:12]><ext>, append a manifest.csv
row with source_folder = this pack's name (so leakage-safe grouped CV treats
it as one distinct pack).

Per user decision: "Bass" becomes a new 19th class; "Rolls" included despite
being fills/patterns rather than one-shots (only files <=3s will ever actually
surface in the app's one-shot pipeline, which is unchanged by this script).

Usage:  venv-clap/bin/python integrate_jiro_pack.py
"""
import csv
import hashlib
import shutil
from pathlib import Path

import soundfile as sf

PACK_DIR = Path("/Users/rayhan/Downloads/Jiro Inagaki & Soul Media - Funky Stuff Pack")
SOURCE_FOLDER = "Jiro Inagaki & Soul Media - Funky Stuff Pack"
LABELED_DIR = Path("training_data/labeled")
MANIFEST = Path("training_data/manifest.csv")
AUDIO_EXT = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def duration_of(path):
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:
        return None


def existing_hashes():
    if not MANIFEST.exists():
        return set()
    with open(MANIFEST) as f:
        return {r["hash"] for r in csv.DictReader(f)}


def main():
    seen = existing_hashes()
    print(f"{len(seen)} hashes already in manifest.csv")

    new_rows = []
    per_label = {}
    skipped_dupe = skipped_bad = 0

    for label_dir in sorted(PACK_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name  # folder name IS the label (matches existing convention)
        files = [f for f in label_dir.iterdir() if f.suffix.lower() in AUDIO_EXT]
        dest_dir = LABELED_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            h = file_hash(f)
            if h in seen:
                skipped_dupe += 1
                continue
            dur = duration_of(f)
            if dur is None:
                skipped_bad += 1
                continue
            dest = dest_dir / f"{h[:12]}{f.suffix.lower()}"
            shutil.copy2(f, dest)
            new_rows.append([h, label, f"{dur:.3f}", f.name, SOURCE_FOLDER, ""])
            seen.add(h)
            per_label[label] = per_label.get(label, 0) + 1

    if new_rows:
        write_header = not MANIFEST.exists()
        with open(MANIFEST, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["hash", "label", "duration", "orig_name", "source_folder", "source_link"])
            w.writerows(new_rows)

    print(f"\nAdded {len(new_rows)} new one-shots, skipped {skipped_dupe} dupes, {skipped_bad} unreadable")
    for label, n in sorted(per_label.items()):
        print(f"  {label:14s} +{n}")


if __name__ == "__main__":
    main()
