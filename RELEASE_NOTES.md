Adds config-driven target profiles for per-app/per-window dictation behavior.

## Download

| Asset | Use on |
|---|---|
| **whisper-dictate-windows-nvidia-setup-0.2.57.exe** | Windows with NVIDIA CUDA |

## Highlights

- `config.json` now supports a `profiles` array.
- Profiles match active window title/process when recording starts.
- Matching profiles can override live-safe settings for that utterance, such as `inject_mode`, `lang`, `initial_prompt`, dictionary settings and audio thresholds.
- Active profile is logged and included in metrics/history events.

## Notes

- Restart-only profile settings such as backend/model/device are detected and reported as requiring restart/model reload.
- First matching profile wins.
