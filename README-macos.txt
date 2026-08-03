PulseMap — macOS build
=======================

Requires Apple Silicon (M1 or later). This build is not signed/notarized, so
macOS Gatekeeper will block the first launch.

SETUP
-----
1. Open the .dmg, drag PulseMap.app to Applications.
2. This build is unsigned (no Apple Developer account), so Gatekeeper blocks
   the first launch. Right-click PulseMap.app -> "Open" -> "Open" again in
   the dialog. (Plain double-click just gets blocked with no way through —
   right-click "Open" is the one-time exception.)
3. After that, launch it normally any way you like.

If macOS still refuses: System Settings -> Privacy & Security -> scroll to
the blocked-app notice -> "Open Anyway".

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
Open Terminal and run:

    /Applications/PulseMap.app/Contents/MacOS/pulsemap

Any error will be printed there. Sending that text back is the most useful
bug report.
