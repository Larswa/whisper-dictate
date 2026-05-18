#!/usr/bin/env python3
r"""voice-pi — all-in-one push-to-talk DICTATION.

Speak prompts instead of typing them. Hold the hotkey, speak softly,
release — the transcribed text is injected into whatever window has
focus (a terminal, a browser chat box, an editor … anything).
A mic→keyboard, not an AI chat: the "AI" is whatever app you're in.

One process: mic capture and Whisper run together, no server, no
network hop. Whisper runs on your NVIDIA GPU (CUDA) when present and
falls back to CPU otherwise — same code, see --device. Use setup.ps1
(Windows) or setup.sh (Linux) for a one-shot, portable install.

First run downloads the model into the Hugging Face cache (turbo
~1.5 GB; large-v3 ~3 GB).

Hold RIGHT CTRL, speak, release → text appears at your cursor.
  --key f9        use a different hold-to-talk key (ctrl_r, alt_r, f9…)
  --key a+b       chord: hold BOTH keys simultaneously (e.g. shift_r+ctrl_r)
  --paste         inject via clipboard + Ctrl+V on X11/Windows
                  (on Wayland direct evdev keycodes are always used instead)
  --no-type       just print what was heard (don't inject — testing)
  --model NAME    Whisper model (default large-v3-turbo, the fastest;
                  env VOICEPI_MODEL)
  --device D      auto|cuda|cpu (default auto; env VOICEPI_DEVICE)
  --lang CODE     spoken-language hint da/en/de/fr… (env VOICEPI_LANG)
                  omit to let Whisper auto-detect (less reliable on short speech)
  --autodetect    alias for omitting --lang

On Wayland (Ubuntu 26.04), text is injected directly via ydotool:
ASCII via ydotool type, æøå via evdev keycodes (compositor maps them
through the DK XKB layout — no clipboard, no paste shortcut).
Stop it by pressing Esc (or Ctrl+C) — that frees the GPU VRAM.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import site
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard


# --- CUDA runtime DLL bootstrap (Windows) -------------------------------
# ctranslate2 (faster-whisper's backend) needs the CUDA runtime libs
# (cublas/cudnn). On Windows the nvidia-*-cu12 pip wheels drop those
# DLLs in site-packages\nvidia\*\bin, which is NOT on the default DLL
# search path. Mirror what LD_LIBRARY_PATH did in the old WSL build:
# register each nvidia\*\bin dir before faster_whisper is imported.
# Guarded + Windows-only so the file still imports cleanly elsewhere.
if os.name == "nt":
    try:
        for sp in site.getsitepackages():
            for d in sorted({os.path.dirname(p) for p in glob.glob(
                    os.path.join(sp, "nvidia", "*", "bin", "*.dll"))}):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:  # noqa: BLE001 — never block startup on this
        print(f"[warn] CUDA DLL bootstrap skipped: {e}", flush=True)

from faster_whisper import WhisperModel  # noqa: E402 — must follow bootstrap

SR = 16000
MODEL_NAME = os.environ.get("VOICEPI_MODEL", "large-v3-turbo")
VALID_DEVICES = ("auto", "cuda", "cpu")
# Compute device: "auto" tries the GPU (CUDA/NVIDIA) and falls back to
# CPU; force with "cuda" or "cpu". faster-whisper/ctranslate2 only
# accelerate on NVIDIA — an AMD GPU box runs CPU. CPU is usable but
# slow, so the default model is the FASTEST (large-v3-turbo).
DEVICE = os.environ.get("VOICEPI_DEVICE", "auto")
# Target loudness (dBFS) quiet input is boosted toward before Whisper
# sees it. Soft voiced speech lands at -35..-45 dBFS where Whisper's
# no-speech gate eats it; normalising to ~-20 recovers it without
# clipping. Lower (e.g. -16) = boost harder.
TARGET_DBFS = float(os.environ.get("VOICEPI_TARGET_DBFS", "-20"))
# Raw-input gate before gain boost. Without this, near-silence gets boosted
# into Whisper's comfort range; with a fixed language hint, Danish silence
# often decodes as a plausible short phrase such as "Tak."
MIN_INPUT_DBFS = float(os.environ.get("VOICEPI_MIN_INPUT_DBFS", "-55"))
MIN_INPUT_SNR_DB = float(os.environ.get("VOICEPI_MIN_SNR_DB", "6"))
# Spoken-language hint. Whisper large-v3(-turbo) is multilingual; a
# fixed hint is far more reliable than auto-detect on short/soft
# utterances (and avoids da+English mixing flip-flop). "da", "en",
# "de", "fr", … ; --autodetect sets this to None (Whisper guesses).
LANG = os.environ.get("VOICEPI_LANG")  # None → Whisper auto-detects
# beam_size=1 is fastest on CPU; raise to 5 for better accuracy at the
# cost of 3-4× slower transcription. VOICEPI_BEAM_SIZE=5 is useful on
# machines without GPU where accuracy matters more than latency.
BEAM_SIZE = int(os.environ.get("VOICEPI_BEAM_SIZE", "1"))
# Optional context hint fed to Whisper before each utterance. Improves
# recognition of domain-specific terms (product names, jargon, names).
# Example: VOICEPI_INITIAL_PROMPT="Winget, whisper-dictate, FactusConsulting"
INITIAL_PROMPT = os.environ.get("VOICEPI_INITIAL_PROMPT") or None


def _resolve_device(want: str) -> tuple[str, str]:
    # → (device, compute_type). "auto" uses the GPU if a CUDA/NVIDIA
    # device is present, else CPU. faster-whisper/ctranslate2 only
    # accelerate on NVIDIA, so an AMD-GPU machine resolves to "cpu"
    # (same as a no-GPU box). int8_float16 on GPU, int8 on CPU.
    want = (want or "auto").lower()
    if want not in VALID_DEVICES:
        raise ValueError(f"invalid device '{want}' (expected: "
                         f"{', '.join(VALID_DEVICES)})")
    if want == "cuda":
        return "cuda", "int8_float16"
    if want == "cpu":
        return "cpu", "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8_float16"
    except Exception:  # noqa: BLE001 — any failure → safe CPU fallback
        pass
    return "cpu", "int8"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="ctrl_r",
                    help="pynput Key name held to talk (ctrl_r, alt_r, f9…) "
                         "or chord: shift_r+ctrl_r")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="Whisper model (default large-v3-turbo, fastest; "
                         "env VOICEPI_MODEL)")
    ap.add_argument("--lang", default=LANG,
                    help="spoken-language hint: da, en, de, fr… "
                         "(env VOICEPI_LANG) — omit to let Whisper auto-detect")
    ap.add_argument("--autodetect", action="store_true",
                    help="explicitly auto-detect language (alias for omitting --lang)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--paste", action="store_const", dest="mode",
                   const="paste",
                   help="inject via clipboard + Ctrl+V on X11/Windows "
                        "(on Wayland direct evdev keycodes are always used)")
    g.add_argument("--no-type", action="store_const", dest="mode",
                   const="print", help="just print, don't inject")
    ap.add_argument("--device", default=DEVICE, choices=VALID_DEVICES,
                    help="auto|cuda|cpu (default auto; env VOICEPI_DEVICE). "
                         "auto = NVIDIA GPU if present, else CPU")
    ap.set_defaults(mode="type")
    return ap


def _noise_snr(a: np.ndarray) -> tuple[float, float]:
    # Percentile-based noise-floor / SNR estimate — no VAD, no deps.
    # Frame the RAW (pre-boost) signal into 30 ms windows; the quiet
    # frames between/around words ARE the noise. Noise floor = 10th
    # pct of per-frame RMS (a real mic property in dBFS); SNR = how
    # far the speech (90th pct) sits above it. SNR is gain-invariant
    # so a uniform boost can't flatter it. Few-frame guard avoids
    # log10(0) on near-empty buffers.
    fr = 480  # 30 ms @ 16 kHz
    n = len(a) // fr
    if n < 4:
        return -90.0, 0.0
    frm = a[:n * fr].reshape(n, fr)
    rms = np.sqrt(np.mean(frm.astype(np.float64) ** 2, axis=1))
    lo = float(np.percentile(rms, 10)) or 1e-9
    hi = float(np.percentile(rms, 90)) or 1e-9
    noise_dbfs = 20 * np.log10(lo)
    snr_db = 20 * np.log10(hi / lo)
    return noise_dbfs, snr_db


def _boost_quiet(a: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(a**2)) or 1e-9)
    cur_dbfs = 20 * np.log10(rms)
    gain = 10 ** ((TARGET_DBFS - cur_dbfs) / 20)
    peak = float(np.max(np.abs(a)) or 1e-9)
    gain = min(gain, 0.99 / peak)  # never clip
    noise_dbfs, snr_db = _noise_snr(a)
    print(f"[cap] raw={cur_dbfs:.0f}dBFS peak={peak:.3f} gain={gain:.1f}x "
          f"noise={noise_dbfs:.0f}dBFS snr={snr_db:.0f}dB", flush=True)
    return (a * gain).astype(np.float32)


def _looks_like_speech(a: np.ndarray) -> tuple[bool, str]:
    rms = float(np.sqrt(np.mean(a**2)) or 1e-9)
    raw_dbfs = 20 * np.log10(rms)
    noise_dbfs, snr_db = _noise_snr(a)
    if raw_dbfs < MIN_INPUT_DBFS:
        return False, (
            f"input too quiet: raw={raw_dbfs:.0f}dBFS "
            f"< {MIN_INPUT_DBFS:.0f}dBFS"
        )
    if snr_db < MIN_INPUT_SNR_DB:
        return False, (
            f"no speech contrast: snr={snr_db:.0f}dB "
            f"< {MIN_INPUT_SNR_DB:.0f}dB"
        )
    return True, (
        f"raw={raw_dbfs:.0f}dBFS noise={noise_dbfs:.0f}dBFS "
        f"snr={snr_db:.0f}dB"
    )


def _transcribe(model: WhisperModel, pcm: np.ndarray,
                lang: str | None) -> str:
    # pcm: int16 mono @ 16 kHz straight from sounddevice — already the
    # rate/layout Whisper wants, so no WAV round-trip or resample (that
    # whole path died with the server). Just int16 → float32 → boost.
    raw_audio = pcm.reshape(-1).astype(np.float32) / 32768.0
    ok, gate = _looks_like_speech(raw_audio)
    if not ok:
        print(f"[gate] {gate}", flush=True)
        return ""
    print(f"[gate] {gate}", flush=True)
    audio = _boost_quiet(raw_audio)
    dur = len(audio) / SR
    in_dbfs = 20 * np.log10(float(np.sqrt(np.mean(audio**2)) or 1e-9))
    t0 = time.monotonic()
    segments, _ = model.transcribe(
        audio,
        language=lang,  # None → Whisper auto-detects
        initial_prompt=INITIAL_PROMPT,  # domain-specific term hints
        beam_size=BEAM_SIZE,
        temperature=[0.0, 0.2],
        # short turns: don't carry prior text — it makes Whisper
        # hallucinate continuations on near-silent input.
        condition_on_previous_text=False,
        # relaxed gates: defaults drop genuinely-quiet-but-real speech
        # as "no speech"; these let soft voiced speech through.
        no_speech_threshold=0.45,
        log_prob_threshold=-1.0,
        vad_filter=True,
        # threshold 0.3 (vs Silero default 0.5): soft voiced speech sits
        # below 0.5 speech-probability. min_silence keeps natural pauses
        # from splitting a sentence mid-thought.
        vad_parameters=dict(threshold=0.3, min_silence_duration_ms=600),
    )
    # Concatenate with Whisper's OWN spacing. Each segment text already
    # carries a leading space on word boundaries (BPE ▁ tokens); a
    # strip()+" ".join() drops that at segment joins → "hørerdig".
    # Join raw, then collapse whitespace runs to one space.
    text = re.sub(r"\s+", " ", "".join(s.text for s in segments)).strip()
    print(f"[stt] dur={dur:.1f}s post-boost={in_dbfs:.0f}dBFS "
          f"compute={time.monotonic() - t0:.1f}s text={text!r}", flush=True)
    return text


_LANG_TO_XKB = {
    "da": "dk", "de": "de", "fr": "fr", "fi": "fi", "sv": "se",
    "nb": "no", "nn": "no", "nl": "nl", "pl": "pl", "pt": "pt",
    "es": "es", "it": "it", "uk": "ua",
}

# Wayland text injection: ydotool type bruger US-keyboard internt.
# Tegn placeret anderledes i ikke-US layouts skal sendes som raw evdev-keycodes
# via ydotool key så compositor anvender det aktive XKB-layout korrekt.
#
# Relevante scancodes (Linux input-event-codes.h):
#   KEY_2=3  KEY_7=8  KEY_MINUS=12  KEY_LEFTBRACE=26
#   KEY_SEMICOLON=39  KEY_APOSTROPHE=40  KEY_LEFTSHIFT=42
#   KEY_COMMA=51  KEY_DOT=52  KEY_SLASH=53

# Tegnsætning der er identisk placeret i alle nordiske + tyske layouts,
# men anderledes end US (f.eks. ? er shift+KEY_MINUS, ikke shift+KEY_SLASH).
_NORDIC_DE_PUNCT: dict[str, list[str]] = {
    '?': ['42:1', '12:1', '12:0', '42:0'],  # shift+KEY_MINUS (US: shift+KEY_SLASH)
    '-': ['53:1', '53:0'],                   # KEY_SLASH       (US: KEY_MINUS)
    '_': ['42:1', '53:1', '53:0', '42:0'],  # shift+KEY_SLASH
    ':': ['42:1', '52:1', '52:0', '42:0'],  # shift+KEY_DOT   (US: shift+KEY_SEMICOLON)
    ';': ['42:1', '51:1', '51:0', '42:0'],  # shift+KEY_COMMA (US: KEY_SEMICOLON)
    '/': ['42:1', '8:1', '8:0', '42:0'],    # shift+KEY_7     (US: KEY_SLASH)
    '"': ['42:1', '3:1', '3:0', '42:0'],    # shift+KEY_2     (US: shift+KEY_APOSTROPHE)
}

# Hjælpefunktioner til at bygge dead-key-sekvenser.
# dead(dk) + plain(lc): fx dead_acute(40) + 'a'(30) → á
def _dead(dk: int, lc: int) -> list[str]:
    return [f'{dk}:1', f'{dk}:0', f'{lc}:1', f'{lc}:0']

def _dead_up(dk: int, lc: int) -> list[str]:
    return [f'{dk}:1', f'{dk}:0', '42:1', f'{lc}:1', f'{lc}:0', '42:0']

def _shift_dead(dk: int, lc: int) -> list[str]:
    return ['42:1', f'{dk}:1', f'{dk}:0', '42:0', f'{lc}:1', f'{lc}:0']

def _shift_dead_up(dk: int, lc: int) -> list[str]:
    return ['42:1', f'{dk}:1', f'{dk}:0', '42:0', '42:1', f'{lc}:1', f'{lc}:0', '42:0']

def _altgr(lc: int) -> list[str]:
    return ['100:1', f'{lc}:1', f'{lc}:0', '100:0']

def _altgr_up(lc: int) -> list[str]:
    return ['100:1', '42:1', f'{lc}:1', f'{lc}:0', '42:0', '100:0']

# Per-layout keycode-kort: XKB-layoutnavn → tegn → ydotool key-sekvens.
# Keycodes 26/39/40 + shift(42) er de nordiske/tyske specialtegntaster —
# samme fysiske placering, forskelligt tegn afhængig af layout.
_LAYOUT_KEYCODES: dict[str, dict[str, list[str]]] = {
    'dk': {  # Dansk: å æ ø
        'å': ['26:1', '26:0'], 'Å': ['42:1', '26:1', '26:0', '42:0'],
        'æ': ['39:1', '39:0'], 'Æ': ['42:1', '39:1', '39:0', '42:0'],
        'ø': ['40:1', '40:0'], 'Ø': ['42:1', '40:1', '40:0', '42:0'],
        **_NORDIC_DE_PUNCT,
    },
    'se': {  # Svensk: å ä ö (ä og ö på samme keycodes som DK's ø og æ)
        'å': ['26:1', '26:0'], 'Å': ['42:1', '26:1', '26:0', '42:0'],
        'ä': ['40:1', '40:0'], 'Ä': ['42:1', '40:1', '40:0', '42:0'],
        'ö': ['39:1', '39:0'], 'Ö': ['42:1', '39:1', '39:0', '42:0'],
        **_NORDIC_DE_PUNCT,
    },
    'de': {  # Tysk: ä ö ü (samme keycodes som nordiske specialtegn)
        'ä': ['40:1', '40:0'], 'Ä': ['42:1', '40:1', '40:0', '42:0'],
        'ö': ['39:1', '39:0'], 'Ö': ['42:1', '39:1', '39:0', '42:0'],
        'ü': ['26:1', '26:0'], 'Ü': ['42:1', '26:1', '26:0', '42:0'],
        **_NORDIC_DE_PUNCT,
    },
    'fi': {  # Finsk: ä ö (ingen å i normal finsk tekst)
        'ä': ['40:1', '40:0'], 'Ä': ['42:1', '40:1', '40:0', '42:0'],
        'ö': ['39:1', '39:0'], 'Ö': ['42:1', '39:1', '39:0', '42:0'],
        **_NORDIC_DE_PUNCT,
    },
    'es': {  # Spansk: ñ direkte; betonede vokaler via dead_acute (AC11=40)
        'ñ': ['39:1', '39:0'], 'Ñ': ['42:1', '39:1', '39:0', '42:0'],
        'á': _dead(40, 30), 'Á': _dead_up(40, 30),
        'é': _dead(40, 18), 'É': _dead_up(40, 18),
        'í': _dead(40, 23), 'Í': _dead_up(40, 23),
        'ó': _dead(40, 24), 'Ó': _dead_up(40, 24),
        'ú': _dead(40, 22), 'Ú': _dead_up(40, 22),
        'ü': _shift_dead(40, 22), 'Ü': _shift_dead_up(40, 22),
    },
    'pt': {  # Portugisisk (EU): ç direkte; vokaler via dead keys
        # AC10(39)=ç  AD12(27)=dead_acute  BKSL(43)=dead_tilde  shift+BKSL=dead_circumflex
        'ç': ['39:1', '39:0'], 'Ç': ['42:1', '39:1', '39:0', '42:0'],
        'á': _dead(27, 30), 'Á': _dead_up(27, 30),
        'é': _dead(27, 18), 'É': _dead_up(27, 18),
        'í': _dead(27, 23), 'Í': _dead_up(27, 23),
        'ó': _dead(27, 24), 'Ó': _dead_up(27, 24),
        'ú': _dead(27, 22), 'Ú': _dead_up(27, 22),
        'à': _shift_dead(27, 30), 'À': _shift_dead_up(27, 30),
        'ã': _dead(43, 30), 'Ã': _dead_up(43, 30),
        'õ': _dead(43, 24), 'Õ': _dead_up(43, 24),
        'â': _shift_dead(43, 30), 'Â': _shift_dead_up(43, 30),
        'ê': _shift_dead(43, 18), 'Ê': _shift_dead_up(43, 18),
        'ô': _shift_dead(43, 24), 'Ô': _shift_dead_up(43, 24),
    },
    'br': {  # Portugisisk (BR): ç direkte; AC11(40)=dead_tilde/dead_circumflex
        'ç': ['39:1', '39:0'], 'Ç': ['42:1', '39:1', '39:0', '42:0'],
        'ã': _dead(40, 30), 'Ã': _dead_up(40, 30),
        'õ': _dead(40, 24), 'Õ': _dead_up(40, 24),
        'â': _shift_dead(40, 30), 'Â': _shift_dead_up(40, 30),
        'ê': _shift_dead(40, 18), 'Ê': _shift_dead_up(40, 18),
        'ô': _shift_dead(40, 24), 'Ô': _shift_dead_up(40, 24),
        # dead_acute via AltGr+AC10(39)
        'á': _dead(39, 30), 'Á': _dead_up(39, 30),  # ydotool type via altgr ikke understøttet
        'é': _dead(39, 18), 'É': _dead_up(39, 18),
        'í': _dead(39, 23), 'Í': _dead_up(39, 23),
        'ó': _dead(39, 24), 'Ó': _dead_up(39, 24),
        'ú': _dead(39, 22), 'Ú': _dead_up(39, 22),
    },
    'pl': {  # Polsk: alle via AltGr+bogstav (KEY_RIGHTALT=100)
        'ą': _altgr(30), 'Ą': _altgr_up(30),  # AltGr+a
        'ę': _altgr(18), 'Ę': _altgr_up(18),  # AltGr+e
        'ó': _altgr(24), 'Ó': _altgr_up(24),  # AltGr+o
        'ś': _altgr(31), 'Ś': _altgr_up(31),  # AltGr+s
        'ź': _altgr(45), 'Ź': _altgr_up(45),  # AltGr+x
        'ż': _altgr(44), 'Ż': _altgr_up(44),  # AltGr+z
        'ć': _altgr(46), 'Ć': _altgr_up(46),  # AltGr+c
        'ń': _altgr(49), 'Ń': _altgr_up(49),  # AltGr+n
        'ł': _altgr(38), 'Ł': _altgr_up(38),  # AltGr+l
    },
    'ua': {  # Ukrainsk: hele det kyrilliske alfabet som direkte keycodes
        # AD-række: й ц у к е н г ш щ з х ї
        'й': ['16:1', '16:0'], 'Й': ['42:1', '16:1', '16:0', '42:0'],
        'ц': ['17:1', '17:0'], 'Ц': ['42:1', '17:1', '17:0', '42:0'],
        'у': ['18:1', '18:0'], 'У': ['42:1', '18:1', '18:0', '42:0'],
        'к': ['19:1', '19:0'], 'К': ['42:1', '19:1', '19:0', '42:0'],
        'е': ['20:1', '20:0'], 'Е': ['42:1', '20:1', '20:0', '42:0'],
        'н': ['21:1', '21:0'], 'Н': ['42:1', '21:1', '21:0', '42:0'],
        'г': ['22:1', '22:0'], 'Г': ['42:1', '22:1', '22:0', '42:0'],
        'ш': ['23:1', '23:0'], 'Ш': ['42:1', '23:1', '23:0', '42:0'],
        'щ': ['24:1', '24:0'], 'Щ': ['42:1', '24:1', '24:0', '42:0'],
        'з': ['25:1', '25:0'], 'З': ['42:1', '25:1', '25:0', '42:0'],
        'х': ['26:1', '26:0'], 'Х': ['42:1', '26:1', '26:0', '42:0'],
        'ї': ['27:1', '27:0'], 'Ї': ['42:1', '27:1', '27:0', '42:0'],
        # AC-række: ф і в а п р о л д ж є
        'ф': ['30:1', '30:0'], 'Ф': ['42:1', '30:1', '30:0', '42:0'],
        'і': ['31:1', '31:0'], 'І': ['42:1', '31:1', '31:0', '42:0'],
        'в': ['32:1', '32:0'], 'В': ['42:1', '32:1', '32:0', '42:0'],
        'а': ['33:1', '33:0'], 'А': ['42:1', '33:1', '33:0', '42:0'],
        'п': ['34:1', '34:0'], 'П': ['42:1', '34:1', '34:0', '42:0'],
        'р': ['35:1', '35:0'], 'Р': ['42:1', '35:1', '35:0', '42:0'],
        'о': ['36:1', '36:0'], 'О': ['42:1', '36:1', '36:0', '42:0'],
        'л': ['37:1', '37:0'], 'Л': ['42:1', '37:1', '37:0', '42:0'],
        'д': ['38:1', '38:0'], 'Д': ['42:1', '38:1', '38:0', '42:0'],
        'ж': ['39:1', '39:0'], 'Ж': ['42:1', '39:1', '39:0', '42:0'],
        'є': ['40:1', '40:0'], 'Є': ['42:1', '40:1', '40:0', '42:0'],
        'ґ': ['43:1', '43:0'], 'Ґ': ['42:1', '43:1', '43:0', '42:0'],
        # AB-række: я ч с м и т ь б ю
        'я': ['44:1', '44:0'], 'Я': ['42:1', '44:1', '44:0', '42:0'],
        'ч': ['45:1', '45:0'], 'Ч': ['42:1', '45:1', '45:0', '42:0'],
        'с': ['46:1', '46:0'], 'С': ['42:1', '46:1', '46:0', '42:0'],
        'м': ['47:1', '47:0'], 'М': ['42:1', '47:1', '47:0', '42:0'],
        'и': ['48:1', '48:0'], 'И': ['42:1', '48:1', '48:0', '42:0'],
        'т': ['49:1', '49:0'], 'Т': ['42:1', '49:1', '49:0', '42:0'],
        'ь': ['50:1', '50:0'], 'Ь': ['42:1', '50:1', '50:0', '42:0'],
        'б': ['51:1', '51:0'], 'Б': ['42:1', '51:1', '51:0', '42:0'],
        'ю': ['52:1', '52:0'], 'Ю': ['42:1', '52:1', '52:0', '42:0'],
    },
}
# Norsk layout er identisk med dansk for æ, ø, å
_LAYOUT_KEYCODES['no'] = _LAYOUT_KEYCODES['dk']


def _build_ydotool_ops(
    text: str,
    keycode_map: dict[str, list[str]],
) -> list[tuple[str, ...]]:
    """Split text into ydotool (subcommand, *args) tuples.

    Characters in keycode_map become ('key', code, ...) events so the
    compositor applies the active XKB layout.  Remaining characters are
    batched into ('type', '--', chunk) calls to minimise process spawns.
    """
    ops: list[tuple[str, ...]] = []
    buf: list[str] = []
    for ch in text:
        if ch in keycode_map:
            if buf:
                ops.append(('type', '--', ''.join(buf)))
                buf = []
            ops.append(('key', *keycode_map[ch]))
        else:
            buf.append(ch)
    if buf:
        ops.append(('type', '--', ''.join(buf)))
    return ops


def _detect_xkb_layout(lang: str | None = None) -> str | None:
    # Priority: VOICEPI_XKB_LAYOUT > XKB_DEFAULT_LAYOUT > /etc/default/keyboard > lang hint
    for var in ("VOICEPI_XKB_LAYOUT", "XKB_DEFAULT_LAYOUT"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    try:
        with open("/etc/default/keyboard") as f:
            for line in f:
                m = re.match(r'XKBLAYOUT="?([^"\s]+)"?', line)
                if m:
                    layout = m.group(1)
                    if layout != "us":  # "us" is often wrong on non-US systems
                        return layout
    except FileNotFoundError:
        pass
    # Fall back: derive layout from spoken-language hint (da→dk, de→de, sv→se…)
    if lang and lang in _LANG_TO_XKB:
        return _LANG_TO_XKB[lang]
    return None


def _find_arecord_device() -> str | None:
    # On PipeWire (Ubuntu 24.04+) PortAudio opens ALSA hardware directly and
    # bypasses PipeWire's mixer — the mic reads as silence. arecord with
    # -D pipewire routes through PipeWire correctly.
    import subprocess, shutil, signal
    if not shutil.which("arecord"):
        return None
    for dev in ("pipewire", "default"):
        try:
            # Start without -d (duration), immediately SIGTERM after 0.3s
            p = subprocess.Popen(
                ["arecord", "-D", dev, "-f", "S16_LE", "-r", "16000", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.3)
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=2)
            stderr = p.stderr.read().decode(errors="replace")
            # SIGTERM gives "Aborted by signal Terminated" — that means it opened OK
            if "Terminated" in stderr or p.returncode in (0, -15, 15):
                return dev
        except Exception:
            pass
    return None


_ARECORD_DEVICE: str | None = None  # set once at startup


class Dictate:
    def __init__(self, model: WhisperModel, key: str, mode: str,
                 lang: str | None):
        global _ARECORD_DEVICE
        self.model = model
        self.key = key
        self.mode = mode  # "type" | "paste" | "print"
        self.lang = lang  # ISO code, or None for auto-detect
        self.frames: list[np.ndarray] = []
        self.recording = False
        self._stream = None
        self._arecord_proc = None
        self._kb = keyboard.Controller()
        self._inject_target_xwin: str | None = None   # XID captured at record start
        self._inject_target_title: str | None = None  # window title for debug log
        xkb = _detect_xkb_layout(lang) or ''
        self._xkb_layout = xkb
        self._keycode_map = _LAYOUT_KEYCODES.get(xkb, {})
        if self._keycode_map:
            print(f"[inject] keycode map: {xkb} ({len(self._keycode_map)} tegn)", flush=True)
        elif bool(os.environ.get('WAYLAND_DISPLAY')):
            print(f"[inject] ingen keycode map for layout '{xkb}' — kun ASCII via ydotool type", flush=True)
        if bool(os.environ.get('WAYLAND_DISPLAY')):
            self._ensure_ydotoold()
        if _ARECORD_DEVICE is None:
            _ARECORD_DEVICE = _find_arecord_device()
        if _ARECORD_DEVICE:
            print(f"[audio] using arecord -D {_ARECORD_DEVICE} (PipeWire route)", flush=True)
        else:
            print("[audio] using sounddevice (direct ALSA)", flush=True)

    def _cb(self, indata, frames, t, status):
        if self.recording:
            self.frames.append(indata.copy())

    def _arecord_reader(self, proc):
        # Read raw S16_LE mono 16kHz from arecord stdout into self.frames
        chunk = SR * 2 * 1  # 1 second of S16 mono = SR*2 bytes
        while self.recording:
            data = proc.stdout.read(chunk // 8)  # read ~125ms chunks
            if not data:
                break
            arr = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
            self.frames.append(arr)

    def _start(self):
        if self.recording:
            return
        self._capture_target_window()
        self.frames = []
        self.recording = True
        if _ARECORD_DEVICE:
            import subprocess
            self._arecord_proc = subprocess.Popen(
                ["arecord", "-D", _ARECORD_DEVICE, "-f", "S16_LE",
                 "-r", str(SR), "-c", "1", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            threading.Thread(target=self._arecord_reader,
                             args=(self._arecord_proc,), daemon=True).start()
        else:
            self._stream = sd.InputStream(
                samplerate=SR, channels=1, dtype="int16", callback=self._cb
            )
            self._stream.start()
        print("● listening…", flush=True)

    def _capture_target_window(self):
        # Capture the active window at the moment PTT is pressed.
        # CPU transcription takes 4+ seconds; by then focus has drifted.
        # Storing the XID lets _inject() refocus before sending Ctrl+V.
        import subprocess, shutil
        self._inject_target_xwin = None
        self._inject_target_title = None
        if not shutil.which("xdotool"):
            return
        try:
            r = subprocess.run(["xdotool", "getactivewindow"],
                               capture_output=True, timeout=1)
            if r.returncode != 0:
                return
            xwin = r.stdout.decode().strip()
            self._inject_target_xwin = xwin
            rt = subprocess.run(["xdotool", "getwindowname", xwin],
                                capture_output=True, timeout=1)
            if rt.returncode == 0:
                self._inject_target_title = rt.stdout.decode().strip()
        except Exception:
            pass

    def _restore_target_focus(self) -> bool:
        # For Wayland-native windows (gedit, ghostty…) xdotool finds an XID
        # via getactivewindow but cannot get the title and cannot reliably
        # activate them — windowactivate returns 0 but focuses an XWayland
        # pseudo-window instead, causing ydotool's Ctrl+V to go there.
        # Skip refocus when the title is unknown; Wayland focus does not
        # drift on its own so the target window should still have it.
        if not self._inject_target_xwin or not self._inject_target_title:
            return False
        import subprocess, shutil
        if not shutil.which("xdotool"):
            return False
        try:
            r = subprocess.run(
                ["xdotool", "windowactivate", "--sync",
                 self._inject_target_xwin],
                capture_output=True, timeout=2)
            return r.returncode == 0
        except Exception:
            return False

    def _wayland_type(self, text: str) -> bool:
        for op in _build_ydotool_ops(text, self._keycode_map):
            if not self._try_ydotool(*op):
                return False
        return True

    def _ensure_ydotoold(self) -> None:
        import subprocess, shutil
        if not shutil.which("ydotoold"):
            return
        if subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True).returncode == 0:
            return
        # Ryd stale socket så ny instans kan binde
        runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        sock = os.path.join(runtime, ".ydotool_socket")
        if os.path.exists(sock):
            try:
                os.remove(sock)
            except OSError:
                pass
        # Foretræk systemd-service — den har XKB_DEFAULT_LAYOUT=dk konfigureret
        r = subprocess.run(["systemctl", "--user", "start", "ydotoold.service"],
                           capture_output=True)
        if r.returncode == 0:
            time.sleep(0.8)
            if subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True).returncode == 0:
                print("[inject] ydotoold startet via systemd", flush=True)
                return
        # Fallback: start direkte med korrekt XKB-env
        env = dict(os.environ)
        if self._xkb_layout:
            env["XKB_DEFAULT_LAYOUT"] = self._xkb_layout
        subprocess.Popen(["ydotoold"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         env=env)
        time.sleep(0.5)
        print(f"[inject] ydotoold startet (XKB={self._xkb_layout or '?'})", flush=True)

    def _try_ydotool(self, *args: str) -> bool:
        import subprocess, shutil
        if not shutil.which("ydotool"):
            return False
        try:
            r = subprocess.run(["ydotool", *args], capture_output=True, timeout=10)
            if r.returncode != 0:
                err = r.stderr.decode(errors="replace").strip()
                if "ydotool_socket" in err:
                    self._ensure_ydotoold()
                    r = subprocess.run(["ydotool", *args],
                                       capture_output=True, timeout=10)
                    err = r.stderr.decode(errors="replace").strip()
                if r.returncode != 0 and err:
                    print(f"[ydotool] {err}", flush=True)
            return r.returncode == 0
        except Exception as e:
            print(f"[ydotool] error: {e}", flush=True)
            return False

    def _inject(self, text: str):
        # Settle: let key-up events reach the compositor before injecting.
        time.sleep(0.4)
        if self.mode == "print":
            print(f"  (heard) {text}", flush=True)
            return
        on_wayland = bool(os.environ.get('WAYLAND_DISPLAY'))

        # CPU transcription takes 4+ seconds — focus has drifted to the
        # terminal by then. Restore the window that was focused when the
        # user pressed the PTT key.
        title = self._inject_target_title or '?'
        if on_wayland and self._restore_target_focus():
            print(f"[inject] → '{title}' (refocused)", flush=True)
            time.sleep(0.1)
        else:
            print(f"[inject] → '{title}'", flush=True)

        if on_wayland:
            # ASCII via ydotool type, æøå via direkte evdev-keycodes (ydotool key).
            # ydotool type v1.0.4 mangler libxkbcommon og dropper non-ASCII stille;
            # men compositor fortolker KEY_LEFTBRACE→å, KEY_SEMICOLON→ø osv. via
            # XKB dk-layout på ydotoold's uinput-enhed — ingen clipboard nødvendig.
            print(f"[inject] ydotool (direkte)", flush=True)
            if not self._wayland_type(text):
                print("[inject] ydotool fejlede — fallback pynput", flush=True)
                self._kb.type(text)
            return

        # X11 / Windows / macOS: paste via clipboard or type per --paste flag.
        if self.mode == "paste":
            import pyperclip
            pyperclip.copy(text)
            self._kb.press(keyboard.Key.ctrl)
            self._kb.press("v")
            self._kb.release("v")
            self._kb.release(keyboard.Key.ctrl)
            return
        self._kb.type(text)

    def _stop_and_transcribe(self):
        if not self.recording:
            return
        self.recording = False
        if self._arecord_proc:
            self._arecord_proc.terminate()
            self._arecord_proc.wait()
            self._arecord_proc = None
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self.frames:
            return
        pcm = np.concatenate(self.frames, axis=0).astype(np.int16)
        if len(pcm) < SR * 0.3:  # <0.3 s — almost certainly a misfire
            print("  (too short — hold the key while you speak)", flush=True)
            return
        try:
            text = _transcribe(self.model, pcm, self.lang)
        except Exception as e:  # noqa: BLE001 — surface any failure
            print(f"  ✗ transcribe error: {e}", flush=True)
            return
        if not text:
            print("  (heard nothing — speak a touch louder / mic closer)",
                  flush=True)
            return
        self._inject(text)

    # pynput key name → evdev key code mapping for common PTT keys
    _EVDEV_MAP = {
        'ctrl_l': 'KEY_LEFTCTRL',   'ctrl_r': 'KEY_RIGHTCTRL',
        'shift_l': 'KEY_LEFTSHIFT', 'shift_r': 'KEY_RIGHTSHIFT',
        'alt_l': 'KEY_LEFTALT',     'alt_r': 'KEY_RIGHTALT',
        'super_l': 'KEY_LEFTMETA',  'super_r': 'KEY_RIGHTMETA',
        **{f'f{i}': f'KEY_F{i}' for i in range(1, 13)},
    }

    def _run_evdev(self, key_names: list[str]):
        # Global hotkey detection via evdev — reads /dev/input/event* directly.
        # Works on pure Wayland where pynput's Xorg backend misses events from
        # Wayland-native windows. Requires user to be in the 'input' group.
        import evdev
        import select

        target_codes: set[int] = set()
        for kn in key_names:
            ecname = self._EVDEV_MAP.get(kn)
            if ecname is None:
                sys.exit(f"unknown key '{kn}' for evdev "
                         f"(supported: {', '.join(self._EVDEV_MAP)})")
            code = getattr(evdev.ecodes, ecname, None)
            if code is None:
                sys.exit(f"evdev has no keycode '{ecname}'")
            target_codes.add(code)

        # Open all input devices that have EV_KEY capability (keyboards)
        devices = []
        for path in evdev.list_devices():
            try:
                d = evdev.InputDevice(path)
                if evdev.ecodes.EV_KEY in d.capabilities():
                    devices.append(d)
            except Exception:
                pass
        if not devices:
            sys.exit("evdev: no keyboard devices found — are you in the 'input' group?")

        pressed: set[int] = set()
        recording = False

        print(f"voice-pi dictation [lang={self.lang or 'auto'}] (evdev). Hold "
              f"[{self.key}] to talk. Ctrl+C to quit.", flush=True)

        try:
            while True:
                r, _, _ = select.select(devices, [], [], 0.5)
                for dev in r:
                    try:
                        events = dev.read()
                    except OSError:
                        continue
                    for ev in events:
                        if ev.type != evdev.ecodes.EV_KEY:
                            continue
                        if ev.code not in target_codes:
                            continue
                        if ev.value == evdev.KeyEvent.key_down:
                            pressed.add(ev.code)
                            if target_codes.issubset(pressed) and not recording:
                                recording = True
                                self._start()
                        elif ev.value == evdev.KeyEvent.key_up:
                            pressed.discard(ev.code)
                            if recording and not target_codes.issubset(pressed):
                                recording = False
                                threading.Thread(
                                    target=self._stop_and_transcribe,
                                    daemon=True).start()
        except KeyboardInterrupt:
            pass
        finally:
            for d in devices:
                try:
                    d.close()
                except Exception:
                    pass
        print("\nbye", flush=True)

    def run(self):
        # Support chord keys: 'shift_r+ctrl_r' means hold both simultaneously.
        # On Wayland: use evdev (reads /dev/input/event* — global, layout-agnostic).
        # On X11: fall back to pynput's xorg backend.
        key_names = [n.strip() for n in self.key.split('+')]

        on_wayland = bool(os.environ.get('WAYLAND_DISPLAY'))
        try:
            import evdev  # noqa: F401
            have_evdev = True
        except ImportError:
            have_evdev = False

        if on_wayland and have_evdev:
            self._run_evdev(key_names)
            return

        # --- pynput fallback (X11 / Windows / macOS) ---
        targets = set()
        for kn in key_names:
            k = getattr(keyboard.Key, kn, None)
            if k is None:
                sys.exit(f"unknown key '{kn}' (e.g. ctrl_r, shift_r, alt_r, f9)")
            targets.add(k)

        pressed: set = set()
        recording = False

        print(f"voice-pi dictation [lang={self.lang or 'auto'}] (pynput). Hold "
              f"[{self.key}] to talk. Esc or Ctrl+C to quit.", flush=True)

        def on_press(k):
            nonlocal recording
            if k == keyboard.Key.esc:
                return False
            pressed.add(k)
            if targets.issubset(pressed) and not recording:
                recording = True
                self._start()

        def on_release(k):
            nonlocal recording
            if k in targets:
                pressed.discard(k)
                if recording and not targets.issubset(pressed):
                    recording = False
                    threading.Thread(target=self._stop_and_transcribe,
                                     daemon=True).start()

        ln = keyboard.Listener(on_press=on_press, on_release=on_release)
        ln.start()
        try:
            while ln.running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            ln.stop()
        print("\nbye", flush=True)


if __name__ == "__main__":
    ap = build_arg_parser()
    a = ap.parse_args()
    lang = None if (a.autodetect or not a.lang) else a.lang

    # Sæt XKB_DEFAULT_LAYOUT fra --lang så ydotool type og evt. auto-startet
    # ydotoold arver det rigtige layout uden manuel konfiguration.
    if lang and not os.environ.get("XKB_DEFAULT_LAYOUT"):
        xkb = _LANG_TO_XKB.get(lang, lang)
        os.environ["XKB_DEFAULT_LAYOUT"] = xkb

    try:
        dev, ctype = _resolve_device(a.device)
    except ValueError as e:
        ap.error(str(e))

    print(f"loading Whisper {a.model} on {dev} ({ctype})… "
          f"first run downloads the model", flush=True)
    if dev == "cpu":
        print("  note: CPU mode — transcription is slower; large-v3-turbo "
              "(default) is the fastest model", flush=True)
    _t = time.monotonic()
    _model = WhisperModel(a.model, device=dev, compute_type=ctype)
    print(f"model ready in {time.monotonic() - _t:.1f}s", flush=True)
    try:
        Dictate(_model, a.key, a.mode, lang).run()
    except KeyboardInterrupt:
        print("\nbye")
