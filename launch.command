#!/bin/bash
# Double-click to launch PulseMap (builds on first run, then just runs).
cd "$(dirname "$0")/native" || exit 1
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
exec cargo run --release --bin pulsemap
