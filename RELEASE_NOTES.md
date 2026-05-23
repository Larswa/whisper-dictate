Local push-to-talk dictation: hold a key, speak softly, release — Whisper transcribes locally and the text is injected into whatever window has focus. No cloud, nothing leaves the machine.

## Which file do I download?

| Asset | Use on | Engine |
|---|---|---|
| **whisper-dictate-windows-nvidia.zip** | Windows + NVIDIA GPU | CUDA (fast, ~1–2 s) |
| **whisper-dictate-windows-cpu.zip** | Windows, no NVIDIA | CPU |
| **whisper-dictate-windows-amd.zip** | Windows with an **AMD** GPU | CPU (AMD GPUs are **not** accelerated — see below) |
| **whisper-dictate-linux-cpu.zip** | Ubuntu 26.04 / 24.04, no NVIDIA | CPU |

Same code in all four — they differ only in the bundled requirements file and launcher. (windows-cpu and windows-amd are identical CPU builds; the AMD one is named so the AMD-box user grabs the obviously-right file.)

## Run it — one click / one command

Unzip, then from the `whisper-dictate/` folder:

- **Windows: double-click `setup.cmd`.** That's the whole install.
  (CLI equivalent: `powershell -ExecutionPolicy Bypass -File setup.ps1`)
- **Linux:** `./setup.sh`

The launcher is idempotent and self-contained: first run installs Python/deps into a machine-local venv, downloads the model, and starts; later runs just start. Hold **Right Ctrl**, speak, release. Press **Esc** (or Ctrl+C) to quit.

Defaults to the fastest model (`large-v3-turbo`) and `VOICEPI_INJECT_MODE=auto`: direct typing for most targets, clipboard paste for known fragile Windows terminal targets. `--device auto` picks the NVIDIA GPU if present, else CPU.

## Known limitations (honest)

- **AMD GPU is not accelerated.** faster-whisper/ctranslate2 only use NVIDIA. The AMD bundle is the CPU build — it does not use the AMD GPU. (GPU accel on AMD would need a different engine, e.g. whisper.cpp + Vulkan — not included.)
- **CPU is slower** (~3–8 s/utterance with turbo) — that's why turbo is the default there.
- **Linux + Wayland:** global hotkeys use evdev and text injection uses ydotool. The user must be in the `input` group for global hotkeys, and supported XKB layouts are listed in the README.
