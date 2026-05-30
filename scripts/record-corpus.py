#!/usr/bin/env python3
"""Record missing benchmark corpus WAV files."""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vp_corpus import load_corpus
from vp_transcribe import SR


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.astype(np.int16).reshape(-1).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmark/corpus.json")
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for item in load_corpus(args.manifest):
        if item.audio.exists() and not args.force:
            continue
        print(f"\n[{item.id}] {item.language} / {item.category}")
        print(item.text)
        input("Press Enter, then speak the sentence...")
        audio = sd.rec(
            int(args.seconds * SR),
            samplerate=SR,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        write_wav(item.audio, audio)
        print(f"saved {item.audio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
