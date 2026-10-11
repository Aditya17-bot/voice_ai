"""End-of-utterance detection using Silero VAD (ONNX, no torch)."""

from collections import deque
from typing import Callable

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

from .audio_io import CHUNK_SAMPLES, SAMPLE_RATE, Microphone

SPEECH_THRESHOLD = 0.5
CHUNK_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE  # 32 ms
TRAILING_SILENCE_MS = 450  # silence that ends an utterance
PRE_ROLL_MS = 320  # audio kept from just before speech starts
MIN_SPEECH_MS = 200  # ignore blips shorter than this
# Background chatter can keep the detector "in speech" for a long time; cut
# the utterance off so transcription stays quick.
MAX_UTTERANCE_MS = 12_000

# Barge-in is stricter than normal listening: the bot's own voice leaks from
# the speakers into the mic, so require higher confidence, sustained longer.
BARGE_IN_THRESHOLD = 0.75
BARGE_IN_SUSTAIN_MS = 320


class UtteranceDetector:
    """Listens on a Microphone and returns one complete spoken utterance."""

    def __init__(self) -> None:
        self._vad = SileroVoiceActivityDetector()
        self.in_speech = False  # the user is mid-utterance (for a visualizer)

    def listen(
        self,
        mic: Microphone,
        carryover: list[np.ndarray] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Block until the user speaks and pauses; return int16 utterance audio.

        `carryover` is speech already captured (e.g. by a barge-in monitor);
        it seeds the utterance so the user's first words aren't lost.
        `should_stop` is polled between mic chunks; when it returns True,
        listening ends and empty audio is returned.
        """
        try:
            return self._listen(mic, carryover, should_stop)
        finally:
            self.in_speech = False

    def _listen(self, mic, carryover, should_stop) -> np.ndarray:
        self._vad.reset()
        pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_MS // CHUNK_MS)
        utterance: list[np.ndarray] = []
        speaking = False
        speech_ms = 0
        silence_ms = 0
        if carryover:
            utterance.extend(carryover)
            speaking = True
            speech_ms = len(carryover) * CHUNK_MS

        while True:
            if should_stop is not None and should_stop():
                return np.zeros(0, dtype=np.int16)
            chunk = mic.read_chunk(timeout=0.1)
            if chunk is None:
                continue
            self.in_speech = speaking
            prob = self._vad(chunk.tobytes())
            is_speech = prob >= SPEECH_THRESHOLD

            if not speaking:
                pre_roll.append(chunk)
                if is_speech:
                    speaking = True
                    utterance.extend(pre_roll)
                    speech_ms = CHUNK_MS
                    silence_ms = 0
                continue

            utterance.append(chunk)
            if len(utterance) * CHUNK_MS >= MAX_UTTERANCE_MS:
                return np.concatenate(utterance)
            if is_speech:
                speech_ms += CHUNK_MS
                silence_ms = 0
            else:
                silence_ms += CHUNK_MS
                if silence_ms >= TRAILING_SILENCE_MS:
                    if speech_ms >= MIN_SPEECH_MS:
                        return np.concatenate(utterance)
                    # Too short to be real speech — reset and keep listening.
                    speaking = False
                    utterance.clear()
                    pre_roll.clear()

    def monitor_barge_in(self, mic, player) -> list[np.ndarray]:
        """Watch the mic while the bot is speaking.

        Returns captured speech chunks if the user barged in (playback is
        stopped), or [] once playback finished undisturbed.
        """
        self._vad.reset()
        streak: list[np.ndarray] = []
        while player.busy:
            chunk = mic.read_chunk()
            if self._vad(chunk.tobytes()) >= BARGE_IN_THRESHOLD:
                streak.append(chunk)
                if len(streak) * CHUNK_MS >= BARGE_IN_SUSTAIN_MS:
                    player.stop()
                    return streak
            else:
                streak.clear()
        return []
