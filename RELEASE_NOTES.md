Adds a reproducible STT benchmark command on top of file transcription.

## Download

| Asset | Use on |
|---|---|
| **whisper-dictate-windows-nvidia-setup-0.2.54.exe** | Windows with NVIDIA CUDA |

## Highlights

- New `--benchmark-files PATH...` command evaluates audio files through backend/model specs.
- `--benchmark-backends` supports specs such as `whisper:large-v3,parakeet:nvidia/parakeet-tdt-0.6b-v3`.
- `--benchmark-jsonl` writes one structured event per file/backend, including backend failures.
- Each benchmark run invokes `--transcribe-file --json` in an isolated child process so model state and backend dependencies do not leak between candidates.

## Notes

- This is a local benchmark foundation for comparing Whisper and Parakeet on the same recordings.
- Cloud backends remain future work and should stay opt-in when added.
