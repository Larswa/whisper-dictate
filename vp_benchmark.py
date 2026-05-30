"""Benchmark/evaluation harness for STT backends."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vp_config import get_value
from vp_corpus import annotate_event, load_corpus, skipped_event


@dataclass(frozen=True)
class BackendSpec:
    raw: str
    backend: str
    model: str | None = None


def parse_backend_specs(spec: str | Iterable[str] | None = None) -> list[BackendSpec]:
    if spec is None:
        spec = get_value("VOICEPI_STT_BACKEND", "whisper") or "whisper"
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",")]
    else:
        parts = [str(p).strip() for p in spec]
    out: list[BackendSpec] = []
    for part in parts:
        if not part:
            continue
        backend, sep, model = part.partition(":")
        backend = backend.strip().lower()
        model = model.strip() if sep else None
        if backend not in ("whisper", "parakeet"):
            raise ValueError(
                f"unsupported benchmark backend {backend!r}; expected whisper or parakeet")
        out.append(BackendSpec(raw=part, backend=backend, model=model or None))
    if not out:
        raise ValueError("at least one benchmark backend is required")
    return out


def _event_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def run_one(
    audio_file: str | Path,
    spec: BackendSpec,
    *,
    python_exe: str = sys.executable,
    app_path: str | Path | None = None,
    base_env: dict[str, str] | None = None,
    timeout_s: int = 900,
) -> dict[str, Any]:
    app = Path(app_path) if app_path else Path(__file__).with_name("voice_pi.py")
    env = dict(os.environ if base_env is None else base_env)
    env["VOICEPI_STT_BACKEND"] = spec.backend
    if spec.model:
        if spec.backend == "parakeet":
            env["VOICEPI_PARAKEET_MODEL"] = spec.model
        else:
            env["VOICEPI_MODEL"] = spec.model
    cmd = [
        python_exe, str(app),
        "--transcribe-file", str(audio_file),
        "--json",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout_s,
    )
    elapsed = time.monotonic() - t0
    event = _event_from_stdout(proc.stdout)
    if event is None:
        event = {
            "event": "benchmark_result",
            "text": "",
            "raw_text": "",
            "source_file": str(audio_file),
        }
    event.update({
        "event": "benchmark_result",
        "benchmark_backend_spec": spec.raw,
        "benchmark_backend": spec.backend,
        "benchmark_model": spec.model,
        "benchmark_elapsed_s": elapsed,
        "benchmark_success": proc.returncode == 0 and bool(event.get("text")),
        "benchmark_returncode": proc.returncode,
    })
    if proc.returncode != 0:
        event["benchmark_error"] = (proc.stderr or proc.stdout).strip()[-4000:]
    return event


def run_benchmark(
    audio_files: Iterable[str | Path] | None,
    backend_specs: str | Iterable[str] | None = None,
    *,
    output_jsonl: str | Path | None = None,
    corpus_manifest: str | Path | None = None,
) -> list[dict[str, Any]]:
    specs = parse_backend_specs(backend_specs)
    results: list[dict[str, Any]] = []
    corpus_items = load_corpus(corpus_manifest) if corpus_manifest else []
    if corpus_items:
        work: list[tuple[str | Path, Any | None]] = [
            (item.audio, item) for item in corpus_items
        ]
    else:
        work = [(path, None) for path in (audio_files or [])]
    if not work:
        raise ValueError("at least one benchmark file or corpus item is required")
    sink = None
    try:
        if output_jsonl:
            out_path = Path(output_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sink = out_path.open("a", encoding="utf-8")
        for audio_file, item in work:
            for spec in specs:
                if item is not None and not Path(audio_file).exists():
                    event = skipped_event(item, "audio file missing")
                else:
                    event = run_one(audio_file, spec)
                    if item is not None:
                        annotate_event(event, item)
                results.append(event)
                line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                if sink:
                    sink.write(line + "\n")
                    sink.flush()
                else:
                    print(line, flush=True)
    finally:
        if sink:
            sink.close()
    return results
