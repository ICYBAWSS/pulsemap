PulseMap — Windows test build
=============================

SETUP
-----
1. Unzip this folder anywhere.
2. Double-click pulsemap.exe.

Windows SmartScreen will warn about an unsigned app — "More info" then
"Run anyway".

If you get an error about VCRUNTIME140.dll or MSVCP140.dll being missing,
install the Microsoft Visual C++ Redistributable (x64) and try again:

    https://aka.ms/vs/17/release/vc_redist.x64.exe

Most machines already have it, so try running
first and only grab this if it complains.

USING IT
--------
- Drop a folder of samples onto the window, or click "Choose folder…".
- Hover a sound to hear it.
- Drag a sound out of the window into a DAW to use it.
- Cmd/Ctrl-drag a sound onto another group to reclassify it.
- Click the "PulseMap" title to go back to the start screen.


IF IT DOESN'T START
-------------------
Open a terminal in this folder and run:

    pulsemap.exe

Any error will be printed there. Sending that text back is the most useful
bug report.
