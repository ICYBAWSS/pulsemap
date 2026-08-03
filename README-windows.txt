PulseMap — Windows test build
=============================

The 117MB audio model is too big to fit in this zip, so install.bat fetches it
the first time you run — after that it just launches the app.

SETUP
-----
1. Unzip this folder anywhere.
2. Double-click install.bat.
   - First run: it downloads the model (~117 MB, once) and starts the app.
   - Every run after: it just starts the app.

Windows SmartScreen will warn about an unsigned app — "More info" then
"Run anyway". That's expected for a build that isn't code-signed.

If you get an error about VCRUNTIME140.dll or MSVCP140.dll being missing,
install the Microsoft Visual C++ Redistributable (x64) and try again:

    https://aka.ms/vs/17/release/vc_redist.x64.exe

Most machines already have it — plenty of apps install it — so try running
first and only grab this if it complains.

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

If install.bat says the download failed, check your internet connection and
re-run it. The model is hosted here if you want to grab it manually and drop it
into the models\ folder:
  https://huggingface.co/icybawss/clap-htsat-unfused-audio-encoder-onnx
