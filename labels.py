"""
Canonical drum-sample categories and the messy folder/file names that map to them.

This is the "folder renaming" brain: when a kit has a folder called "BD", "Kicks",
or "kik", they all become the canonical label "Kick". Edit this freely -- add
aliases you see in real kits, add/remove categories, or re-order.

IMPORTANT DISTINCTION -- specific type vs. grab-bag:
  A flat "Kicks" folder is training gold: "Kick" is one acoustic thing.
  A flat "Percs" folder is NOT: it's rimshot + shaker + tambourine + cowbell +
  whatever else, piled together. Labeling all of it "Perc" teaches the trainer
  a muddy blob, and worse, it CONTRADICTS the good labels -- a shaker sitting in
  someone's "Percs" folder would get called "Perc" while a shaker from another
  kit's "Shaker/" folder gets called "Shaker". Same sound, two labels, both
  classes get worse.

  So there is no generic "Perc" or "FX" class here. Only SPECIFIC acoustic types
  are trainable labels (Rimshot, Shaker, Tambourine, Conga, Cowbell, Woodblock,
  Snap, ...). A file only gets one of those if a specific keyword matches --
  either in the filename, or in a specific-named folder.

  Files that live in a catch-all folder (Percs, One Shots, Drums, Misc, FX,
  Samples, Sounds, Assorted...) with no specific keyword anywhere fall through
  to CATCHALL_PATTERNS and get routed to the UNSORTED holding bucket instead of
  a class -- see collect.py. Those still count toward the map (unsupervised
  layout doesn't need labels) and still help the coarse "is this percussion"
  boundary, but they're excluded from classifier training until sorted.
  Later, once the trainer is solid on the clean specific classes, it can be
  pointed back at the unsorted pile to auto-sort it -- but only works if the
  grab-bag never touched training in the first place.

Rules:
- Order matters. The FIRST matching SPECIFIC label wins, so put SPECIFIC labels
  above GENERAL ones (e.g. "Open Hat" before "Closed Hat", "Rimshot" before
  "Snare", "808" before "Kick"). Catch-all patterns are checked only after ALL
  specific patterns have failed to match.
- Aliases are regular expressions, matched case-insensitively against the last
  two path components (the immediate folder + the filename), so the outer kit
  name (e.g. "Trap Snare Kit vol 3") doesn't mislabel everything.
- \\b means a word boundary, so r"\\bhh\\b" matches "hh" but not "ohh".
"""
import os
import re

# Ordered: specific -> general. NOTE: no "Perc" or "FX" grab-bag entries here --
# see the module docstring for why.
CATEGORIES = [
    ("Open Hat",   [r"open.?hat", r"open.?hh", r"\boh\b", r"\bohh\b", r"\bohat\b"]),
    ("Closed Hat", [r"closed.?hat", r"\bchh\b", r"hi.?hat", r"\bhats?\b", r"\bhh\b"]),
    ("808",        [r"\b808s?\b", r"sub.?bass", r"\bsub\b"]),
    ("Kick",       [r"\bkicks?\b", r"\bkick", r"\bbd\b", r"\bkik", r"\bkck\b"]),
    ("Snare",      [r"\bsnares?\b", r"\bsnare", r"\bsd\b", r"\bsnr\b"]),
    ("Clap",       [r"\bclaps?\b", r"\bclap", r"\bclp\b"]),
    ("Snap",       [r"\bsnaps?\b", r"finger.?snap"]),
    ("Rimshot",    [r"\brimshot", r"\brim\b", r"rim.?shot", r"\brs\b"]),
    ("Crash",      [r"\bcrash", r"\bsplash"]),
    ("Ride",       [r"\bride\b"]),
    ("Cymbal",     [r"\bcymbals?\b", r"\bcym\b"]),
    ("Tom",        [r"\btoms?\b"]),
    ("Shaker",     [r"\bshakers?\b", r"\bshkr\b"]),
    ("Tambourine", [r"tambou?rine", r"\btambs?\b", r"\btam\b"]),
    ("Conga",      [r"\bcongas?\b", r"\bbongos?\b"]),
    ("Cowbell",    [r"cowbell", r"\bcow\b"]),
    ("Woodblock",  [r"wood.?block", r"rim.?click", r"\bclick\b", r"\bwb\b"]),
    ("Vocal",      [r"\bvocals?\b", r"\bvox\b", r"ad.?libs?", r"\bchants?\b"]),
]

# Grab-bag folder/file names: NOT specific enough to be a training label. Files
# that only match here (no specific pattern above matched first) are routed to
# the unsorted holding bucket. Add more aliases as you spot them in real kits.
CATCHALL_PATTERNS = [
    r"\bpercs?\b", r"\bpercussion\b",
    r"\bone.?shots?\b",
    r"\bdrums?\b",
    r"\bmisc(ellaneous)?\b",
    r"\bfx\b", r"\beffects?\b",
    r"\bsamples?\b", r"\bsounds?\b",
    r"\bassorted\b", r"\bvarious\b", r"\bother\b", r"\bmixed\b", r"\bloose\b",
]

# Sentinel returned by classify_path for catch-all matches. Not in LABELS, so
# it never becomes a training class.
UNSORTED = "_unsorted"

# Pre-compile for speed.
_COMPILED = [(label, [re.compile(p, re.IGNORECASE) for p in pats])
             for label, pats in CATEGORIES]
_COMPILED_CATCHALL = [re.compile(p, re.IGNORECASE) for p in CATCHALL_PATTERNS]

# All canonical *trainable* labels, in order. Deliberately excludes UNSORTED.
LABELS = [label for label, _ in CATEGORIES]


def classify_path(path):
    """Classify an audio file. Returns one of:
      - a specific label (e.g. "Kick", "Rimshot")  -> training gold
      - labels.UNSORTED                            -> grab-bag, hold out
      - None                                       -> no drum-related match, skip

    Only the immediate parent folder + filename are considered, so a kit named
    "Snare Bundle" won't stamp every sound as Snare. Specific patterns are
    always checked before catch-all patterns, so a shaker sitting inside a
    "Percs" folder still gets typed as "Shaker" if its filename says so.
    """
    parent = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)
    hay = f"{parent} {name}".replace("_", " ").replace("-", " ")

    for label, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(hay):
                return label

    for pat in _COMPILED_CATCHALL:
        if pat.search(hay):
            return UNSORTED

    return None
