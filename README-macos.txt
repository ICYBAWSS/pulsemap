PulseMap — macOS build
=======================

Requires Apple Silicon (M1 or later). This build is not signed/notarized, so
macOS Gatekeeper will block the first launch.

SETUP
-----
1. Open the .dmg, drag PulseMap.app to Applications.
2. Try to open it — Gatekeeper will block it (unsigned, no Apple Developer
   account). Go to System Settings -> Privacy & Security -> scroll down to
   the blocked-app notice -> "Open Anyway". Launch it once more from
   Applications or Spotlight and it'll go through.
3. If "Open Anyway" doesn't appear, right-click PulseMap.app -> "Open" ->
   "Open" again in the dialog. If that still doesn't work, open Terminal
   and run:

       xattr -dr com.apple.quarantine /Applications/PulseMap.app

After the first successful launch, open it normally any way you like.

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
