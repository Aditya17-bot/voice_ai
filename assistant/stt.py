"""Speech-to-text via faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel

# Not used: hotwords / initial_prompt. They fixed "search your tube" but made
# silence take ~23 s to transcribe and turned loud noise into "www.cf.com".
# Segments that are probably not speech are already skipped by faster-whisper
# (no_speech_threshold 0.6 with log_prob_threshold -1.0).


class SpeechToText:
    # "base.en" (English-only) is as fast as multilingual "base" and got names
    # right that base missed ("Clash Royale", "lo-fi"). "small.en" was ~3x
    # slower on this CPU, which blows the 2 s budget.
    def __init__(self, model_size: str = "base.en") -> None:
        self._model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_int16: np.ndarray) -> str:
        """Transcribe a 16 kHz int16 mono utterance. Returns stripped text."""
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
