@echo off
REM One-click setup for the PulseMap Windows test build: fetch the 117 MB audio
REM model (once) and launch the app. curl ships with Windows 10+ so there is
REM nothing to install first.
setlocal
cd /d "%~dp0"

set "MODEL=models\audio_model.onnx"
set "URL=https://github.com/ICYBAWSS/pulsemap/releases/download/v0.1/audio_model.onnx"

if not exist "%MODEL%" (
  echo Downloading the audio model. This happens once, ~117 MB...
  curl -fL -o "%MODEL%" "%URL%"
  if errorlevel 1 (
    echo.
    echo Download failed. Check your internet connection and re-run this file.
    if exist "%MODEL%" del "%MODEL%"
    pause
    exit /b 1
  )
)

start "" "pulsemap.exe"
