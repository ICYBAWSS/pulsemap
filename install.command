#!/bin/bash
# One-click setup for the macOS test build: fetch the 117 MB audio model
# (once) and launch the app. Double-click this file in Finder.
set -e
cd "$(dirname "$0")"

MODEL="models/audio_model.onnx"
URL="https://huggingface.co/icybawss/clap-htsat-unfused-audio-encoder-onnx/resolve/main/audio_model.onnx"

if [ ! -f "$MODEL" ]; then
  echo "Downloading the audio model. This happens once, ~117 MB..."
  if ! curl -fL -o "$MODEL" "$URL"; then
    echo ""
    echo "Download failed. Check your internet connection and re-run this file."
    rm -f "$MODEL"
    read -p "Press return to close..."
    exit 1
  fi
fi

xattr -dr com.apple.quarantine pulsemap 2>/dev/null || true
exec ./pulsemap
