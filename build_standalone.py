#!/usr/bin/env python3
"""
Build a SINGLE self-contained PulseMap.html the user double-clicks -- no server,
no Deno/Python/Node, nothing installed. It opens off file:// in any browser.

How it stays file://-friendly:
  - No `<script type="module">` (browsers block module scripts on file://).
    Everything runs inside one plain <script> in an async IIFE.
  - ESM-only libs (transformers.js, umap-js) are loaded via DYNAMIC import() of
    https CDN URLs -- which file:// pages ARE allowed to do.
  - d3 + force-graph are UMD globals via <script src=https> (fine on file://).
  - No local fetch(): prototypes.json is inlined as a JS object, and audio is
    read from dropped File objects (no origin issue).

Run:  python3 build_standalone.py   ->  PulseMap.html
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "PulseMap.html"


def strip_exports(js):
    return re.sub(r'^\s*export\s+', '', js, flags=re.MULTILINE)


def main():
    classifier = strip_exports((HERE / "classifier.js").read_text())

    pipeline = (HERE / "pipeline.js").read_text()
    # Drop the three module imports (deps are provided in-scope by the wrapper).
    pipeline = "\n".join(
        l for l in pipeline.splitlines()
        if not l.strip().startswith("import ")
    )
    pipeline = strip_exports(pipeline)
    # Replace the prototypes fetch with the inlined global.
    pipeline = pipeline.replace(
        "    const res = await fetch('prototypes.json');\n    _prototypes = await res.json();",
        "    _prototypes = PULSEMAP_PROTOTYPES;",
    )

    app = (HERE / "app.js").read_text()
    app = "\n".join(
        l for l in app.splitlines()
        if not l.strip().startswith("import ")
    )

    prototypes = json.loads((HERE / "prototypes.json").read_text())

    # Pull the <style> + <body> markup out of index.html (keep it the single
    # source of truth for the UI), but strip its module <script> tag.
    index = (HERE / "index.html").read_text()
    style = re.search(r"<style>.*?</style>", index, re.DOTALL).group(0)
    body_inner = re.search(r"<body>(.*?)</body>", index, re.DOTALL).group(1)
    body_inner = re.sub(r'<script[^>]*src="app\.js"[^>]*></script>', "", body_inner)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PulseMap</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="https://cdn.jsdelivr.net/npm/force-graph"></script>
{style}
</head>
<body>
{body_inner}
<script>
// === PulseMap, fully self-contained. No server, no install. =================
(async function bootPulseMap() {{
  const dbg = document.getElementById('drop');
  try {{
    // ESM-only deps via dynamic import (allowed from file://).
    const T = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3');
    const AutoProcessor = T.AutoProcessor;
    const ClapAudioModelWithProjection = T.ClapAudioModelWithProjection;
    const {{ UMAP }} = await import('https://cdn.jsdelivr.net/npm/umap-js@1/+esm');
    const PULSEMAP_PROTOTYPES = {json.dumps(prototypes)};

    // ---- classifier.js ----
{classifier}

    // ---- pipeline.js ----
{pipeline}

    // ---- app.js ----
{app}
  }} catch (e) {{
    console.error(e);
    if (dbg) dbg.innerHTML =
      '<div style="max-width:520px;text-align:center;color:#e6e8ec;font-family:sans-serif">'
      + '<h2>Couldn\\'t start</h2><p style="color:#8a929e">'
      + 'This needs an internet connection on first run (to fetch the sound-recognition model). '
      + 'Error: ' + (e && e.message) + '</p></div>';
  }}
}})();
</script>
</body>
</html>
"""
    OUT.write_text(html)
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT} ({size_mb:.1f} MB) — double-click to run, no install needed.")


if __name__ == "__main__":
    main()
