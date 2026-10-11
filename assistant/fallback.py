"""Fallback engine: fillers while the LLM is slow, canned replies when it fails.

Keeps the user engaged instead of leaving silence or emitting error messages.
"""

import random
import threading
import wave
from pathlib import Path

import numpy as np

from .audio_io import Player
from .tts import TextToSpeech

FILLERS_DIR = Path(__file__).resolve().parent.parent / "fillers"

# Spoken while the LLM is still working on its first sentence.
FILLER_PHRASES = [
    "Hmm, let me think about that.",
    "Good question. Give me a second.",
    "Okay, one moment.",
    "Let me work that out.",
    "Right, thinking about that now.",
]

# Spoken when the LLM errors out — conversational, never an error message.
RECOVERY_PHRASES = [
    "You know, I lost my train of thought there. Could you say that again?",
    "Sorry, my mind wandered for a moment. Ask me once more?",
    "I didn't quite gather my thoughts on that one. Mind repeating it?",
]

FIRST_FILLER_AFTER_S = 1.2  # silence gap before the first filler
NEXT_FILLER_EVERY_S = 4.0  # keep-alive cadence if the LLM is still quiet


def _wav_path(index: int) -> Path:
    return FILLERS_DIR / f"filler_{index:02d}.wav"


def _save_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(audio.tobytes())


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return audio, rate


class FallbackEngine:
    """Pre-synthesized filler clips + a watchdog that plays them during gaps."""

    def __init__(self, tts: TextToSpeech, player: Player) -> None:
        self._player = player
        self._tts = tts
        self._fillers = self._load_or_generate_fillers(tts)
        self._lock = threading.Lock()  # _pick state is touched from watchdog threads too
        self._last_filler: int | None = None
        self._last_recovery: int | None = None

    def _load_or_generate_fillers(
        self, tts: TextToSpeech
    ) -> list[tuple[np.ndarray, int]]:
        FILLERS_DIR.mkdir(parents=True, exist_ok=True)
        fillers = []
        for i, phrase in enumerate(FILLER_PHRASES):
            path = _wav_path(i)
            if not path.exists():
                audio, rate = tts.synthesize(phrase)
                _save_wav(path, audio, rate)
            fillers.append(_load_wav(path))
        return fillers

    def _pick(self, count: int, last: int | None) -> int:
        choices = [i for i in range(count) if i != last]
        if not choices:  # single clip configured — repeat is the only option
            return 0
        return random.choice(choices)

    def play_filler(self) -> None:
        """Queue a filler clip, never the same one twice in a row."""
        with self._lock:
            index = self._pick(len(self._fillers), self._last_filler)
            self._last_filler = index
        audio, rate = self._fillers[index]
        self._player.enqueue(audio, rate)

    def speak_recovery(self) -> None:
        """Spoken reply for when the LLM failed — synthesized live, rotated."""
        with self._lock:
            index = self._pick(len(RECOVERY_PHRASES), self._last_recovery)
            self._last_recovery = index
        try:
            audio, rate = self._tts.synthesize(RECOVERY_PHRASES[index])
            self._player.enqueue(audio, rate)
        except Exception:  # noqa: BLE001 — if even TTS is down, silence beats a crash
            print("   (TTS unavailable for recovery phrase)")

    def watchdog(self) -> "Watchdog":
        return Watchdog(self)


class Watchdog:
    """Plays fillers while waiting for the LLM's first sentence.

    Usage:
        with fallback.watchdog():   # starts the timer
            ... first sentence arrives -> context exit cancels it
    """

    def __init__(self, engine: FallbackEngine) -> None:
        self._engine = engine
        self._done = threading.Event()
        # Held while deciding to play a filler, so one can't start after cancel().
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        wait = FIRST_FILLER_AFTER_S
        while not self._done.wait(wait):
            with self._lock:
                if self._done.is_set():
                    return
                self._engine.play_filler()
            wait = NEXT_FILLER_EVERY_S

    def __enter__(self) -> "Watchdog":
        self._thread.start()
        return self

    def cancel(self) -> None:
        with self._lock:
            self._done.set()

    def __exit__(self, *exc) -> None:
        self.cancel()
