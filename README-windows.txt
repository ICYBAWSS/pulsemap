PulseMap — Windows test build
=============================

One extra step: the 117MB audio model is too big for GitHub's file limit, so it
is downloaded separately rather than being inside this zip.

SETUP
-----
1. Unzip this folder anywhere.
2. Download "audio_model.onnx" from the project's Releases page.
3. Put it in the "models" folder next to pulsemap.exe, so you have:

     pulsemap.exe
     onnxruntime.dll  (and any other .dll from the zip)
     models\
       audio_model.onnx     <- the one you just downloaded
       model.json
       mel_slaney.npy

4. Run pulsemap.exe.

Windows SmartScreen will warn about an unsigned app — "More info" then
"Run anyway". That's expected for a build that isn't code-signed.

USING IT
--------
- Drop a folder of samples onto the window, or click "Choose folder…".
- Hover a sound to hear it.
- Drag a sound out of the window into a DAW to use it.
- Cmd/Ctrl-drag a sound onto another group to reclassify it.
- Click the "PulseMap" title to go back to the start screen.

Everything is analyzed locally. No audio leaves the machine.

IF IT DOESN'T START
-------------------
Open a terminal in this folder and run:

    pulsemap.exe

Any error will be printed there. Sending that text back is the most useful
bug report.
