"""N64-style synthesized sound effects for the TUI.

Playback strategy (tried in order):
  1. macOS  → afplay (CoreAudio, works inside tmux/SSH-local)
  2. Other  → sounddevice (if installed and a device exists)
  3. Always → terminal BEL as last resort
"""
from __future__ import annotations

import io
import subprocess
import sys
import wave as wavlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# sounddevice is optional and only used on non-macOS
_sd = None
_SD_AVAILABLE = False
if sys.platform != "darwin":
    try:
        import sounddevice as _sd  # type: ignore
        _sd.query_devices(kind="output")
        _SD_AVAILABLE = True
    except Exception:
        pass

SAMPLE_RATE = 44100
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio")


# ── Waveform helpers ─────────────────────────────────────────────────────────

def _adsr(n: int, attack=0.01, decay=0.05, sustain=0.7, release=0.1) -> np.ndarray:
    env = np.ones(n)
    a, d, r = int(n * attack), int(n * decay), int(n * release)
    s = n - a - d - r
    if a > 0: env[:a] = np.linspace(0, 1, a)
    if d > 0: env[a:a+d] = np.linspace(1, sustain, d)
    if s > 0: env[a+d:a+d+s] = sustain
    if r > 0: env[a+d+s:] = np.linspace(sustain, 0, n - (a+d+s))
    return env


def _square(freq: float, duration: float, volume: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = np.sign(np.sin(2 * np.pi * freq * t))
    env = _adsr(len(wave), attack=0.005, decay=0.05, sustain=0.7, release=0.1)
    return (wave * env * volume).astype(np.float32)


def _sequence(*notes: tuple[float, float], gap: float = 0.01) -> np.ndarray:
    silence = np.zeros(int(SAMPLE_RATE * gap), dtype=np.float32)
    parts = []
    for freq, dur in notes:
        parts.append(_square(freq, dur))
        parts.append(silence)
    return np.concatenate(parts)


# ── Playback ─────────────────────────────────────────────────────────────────

def _to_wav(wave: np.ndarray) -> bytes:
    data = (wave * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wavlib.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


def _play(wave: np.ndarray, bells: int) -> None:
    if sys.platform == "darwin":
        try:
            proc = subprocess.Popen(
                ["afplay", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc.communicate(_to_wav(wave))
            return
        except Exception:
            pass
    if _SD_AVAILABLE:
        try:
            _sd.play(wave, samplerate=SAMPLE_RATE, blocking=True)
            return
        except Exception:
            pass
    # Last resort: terminal bell
    sys.stderr.write("\a" * bells)
    sys.stderr.flush()


def _play_async(wave: np.ndarray, bells: int = 1) -> None:
    _executor.submit(_play, wave, bells)


# ── Note frequencies ─────────────────────────────────────────────────────────
C4, E4, G4, A4 = 261.63, 329.63, 392.00, 440.00
C5, D5, E5, G5, A5, B5 = 523.25, 587.33, 659.25, 784.00, 880.00, 987.77
C6, E6 = 1046.50, 1318.51


# ── Public sound effects ──────────────────────────────────────────────────────

def play_nav_click() -> None:
    _play_async(_square(A5, 0.04, volume=0.18), bells=1)

def play_favorite() -> None:
    _play_async(_sequence((C5, 0.06), (E6, 0.12)), bells=2)

def play_gen_start() -> None:
    _play_async(_sequence((G4, 0.10), (C5, 0.16)), bells=1)

def play_gen_complete() -> None:
    _play_async(_sequence((C5, 0.07), (E5, 0.07), (G5, 0.07), (C6, 0.22)), bells=1)

def play_masterpiece() -> None:
    _play_async(
        _sequence((E5, 0.07), (G5, 0.07), (A5, 0.07), (C6, 0.10), (A5, 0.07), (C6, 0.30)),
        bells=3,
    )

def play_train_start() -> None:
    _play_async(_sequence((G4, 0.12), (D5, 0.18)), bells=1)

def play_train_end() -> None:
    _play_async(_sequence((G4, 0.08), (C5, 0.08), (E5, 0.28)), bells=2)

def play_save() -> None:
    _play_async(_sequence((C5, 0.07), (C6, 0.10)), bells=1)

def play_resume() -> None:
    _play_async(_sequence((E4, 0.10), (G4, 0.10), (C5, 0.18)), bells=1)

def play_startup() -> None:
    _play_async(
        _sequence((C5, 0.08), (E5, 0.08), (G5, 0.08), (C6, 0.08), (G5, 0.08), (E5, 0.08), (C6, 0.28)),
        bells=2,
    )
