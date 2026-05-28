"""Custom vocabulary and deterministic post-transcription replacements."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() not in ("", "0", "false", "no", "off")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[dictionary] ignoring invalid {name}={raw!r}", flush=True)
        return default


def _default_path() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "WhisperDictate" / "dictionary.json"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "whisper-dictate" / "dictionary.json"


def _candidate_paths() -> list[Path]:
    raw = os.environ.get("VOICEPI_DICTIONARY")
    if raw:
        return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]

    here = Path(__file__).resolve().parent
    return [
        _default_path(),
        here / "dictionary.json",
        here / "dictionary.txt",
    ]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = str(item).strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _parse_mapping_line(line: str) -> tuple[str, str] | None:
    for sep in ("=>", "->", "="):
        if sep in line:
            left, right = line.split(sep, 1)
            left = left.strip().strip("\"'")
            right = right.strip().strip("\"'")
            if left and right:
                return left, right
    if ":" in line:
        left, right = line.split(":", 1)
        left = left.strip().strip("\"'")
        right = right.strip().strip("\"'")
        if left and right:
            return left, right
    return None


def _parse_text_config(text: str) -> tuple[list[str], dict[str, str]]:
    terms: list[str] = []
    replacements: dict[str, str] = {}
    section = "terms"

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = line.rstrip(":").strip().lower()
        if header in ("[terms]", "terms"):
            section = "terms"
            continue
        if header in ("[replacements]", "replacements"):
            section = "replacements"
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if section == "replacements":
            mapping = _parse_mapping_line(line)
            if mapping:
                replacements[mapping[0]] = mapping[1]
            continue
        terms.append(line.strip("\"'"))
    return terms, replacements


def _load_path(path: Path) -> tuple[list[str], dict[str, str]]:
    data = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(data)
        terms_raw = obj.get("terms", [])
        terms: list[str] = []
        for item in terms_raw:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict) and item.get("term"):
                terms.append(str(item["term"]))
        repl_raw = obj.get("replacements", {})
        replacements: dict[str, str] = {}
        if isinstance(repl_raw, dict):
            replacements = {str(k): str(v) for k, v in repl_raw.items()}
        elif isinstance(repl_raw, list):
            for item in repl_raw:
                if isinstance(item, dict) and item.get("from") and item.get("to"):
                    replacements[str(item["from"])] = str(item["to"])
        return terms, replacements
    return _parse_text_config(data)


@dataclass
class Dictionary:
    terms: list[str] = field(default_factory=list)
    replacements: dict[str, str] = field(default_factory=dict)
    paths: list[Path] = field(default_factory=list)

    def prompt_terms(self) -> list[str]:
        max_terms = _int_env("VOICEPI_DICTIONARY_MAX_TERMS", 80)
        max_chars = _int_env("VOICEPI_DICTIONARY_PROMPT_CHARS", 1200)
        out: list[str] = []
        chars = 0
        for term in self.terms:
            added = len(term) + (2 if out else 0)
            if len(out) >= max_terms or chars + added > max_chars:
                break
            out.append(term)
            chars += added
        return out

    def build_prompt(self, base_prompt: str | None) -> str | None:
        terms = self.prompt_terms()
        parts = []
        if base_prompt:
            parts.append(base_prompt.strip())
        if terms:
            parts.append("Vocabulary: " + ", ".join(terms))
        return "\n".join(p for p in parts if p) or None

    def apply_replacements(self, text: str) -> tuple[str, list[dict[str, object]]]:
        if not text or not self.replacements:
            return text, []
        changed: list[dict[str, object]] = []
        out = text
        for src, dst in sorted(self.replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
            if not src:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE)
            out, count = pattern.subn(dst, out)
            if count:
                changed.append({"from": src, "to": dst, "count": count})
        return out, changed


def load_dictionary() -> Dictionary:
    if not _truthy(os.environ.get("VOICEPI_DICTIONARY_ENABLED", "1")):
        return Dictionary()

    terms: list[str] = []
    replacements: dict[str, str] = {}
    loaded: list[Path] = []
    for path in _candidate_paths():
        if not path.exists():
            continue
        try:
            p_terms, p_replacements = _load_path(path)
        except Exception as e:  # noqa: BLE001 - config errors should not block dictation
            print(f"[dictionary] could not load {path}: {e}", flush=True)
            continue
        terms.extend(p_terms)
        replacements.update(p_replacements)
        loaded.append(path)

    dictionary = Dictionary(_dedupe(terms), replacements, loaded)
    if loaded:
        print(
            f"[dictionary] loaded {len(dictionary.terms)} terms and "
            f"{len(dictionary.replacements)} replacements from "
            f"{', '.join(str(p) for p in loaded)}",
            flush=True,
        )
    return dictionary


DICTIONARY = load_dictionary()
