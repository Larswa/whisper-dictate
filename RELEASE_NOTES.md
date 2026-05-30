Windows-focused release with the unified Settings UI, Parakeet tuning, and safer injection behavior.

## Download

| Asset | Use on |
|---|---|
| **whisper-dictate-windows-nvidia-setup-0.2.52.exe** | Windows with NVIDIA CUDA |

## Highlights

- Settings UI now includes clickable and hoverable `?` help on Core, Quality, Dictionary, and Output settings.
- Added `VOICEPI_RELEASE_TAIL_MS` to keep recording briefly after hotkey release and avoid clipping final words.
- Parakeet model dropdown is trimmed to the practical choices: multilingual v3, pure-English TDT 1.1B, and fast English-only v2.
- Parakeet/NeMo startup and progress noise is hidden unless `VOICEPI_STT_DEBUG=1`.
- English contractions and other layout-sensitive punctuation are injected safely on Danish Windows keyboard layouts.

## Notes

- Parakeet startup can take tens of seconds on Windows because NeMo/PyTorch model restore is heavy. Once loaded, inference remains very fast.
- `VOICEPI_LANG` is a Whisper-only language hint; Parakeet v3 autodetects language.
- Dictionary replacements are applied after transcription for both Whisper and Parakeet.
