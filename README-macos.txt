PulseMap — macOS build
=======================

Requires Apple Silicon (M1 or later). This build is not signed/notarized, so
macOS Gatekeeper will block the first launch.

SETUP
-----
1. Unzip this folder anywhere.
2. Right-click "install.command" -> "Open" -> "Open" again in the dialog.
   (Double-clicking directly will just get blocked with no way to proceed —
   right-click "Open" is the one-time exception that lets it run.)
3. It downloads the audio model (~117 MB, once) and launches the app.

Later launches: right-click "install.command" -> "Open" once more, or just
double-click "pulsemap" directly (it only needs the Gatekeeper exception once
per file).

If macOS still refuses to open it: System Settings -> Privacy & Security ->
scroll down to the blocked-app notice -> "Open Anyway".

USING IT
--------
- Drop a folder of samples onto the window, or click "Choose folder…".
- Hover a sound to hear it.
- Drag a sound out of the window into a DAW to use it.
- Cmd-drag a sound onto another group to reclassify it.
- Click the "PulseMap" title to go back to the start screen.

Everything is analyzed locally. No audio leaves the machine.

IF IT DOESN'T START
--------------------
Open Terminal, cd into this folder, and run:

    ./pulsemap

Any error will be printed there. Sending that text back is the most useful
bug report.
