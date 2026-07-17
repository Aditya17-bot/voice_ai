"""Speech-to-text via faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel


class SpeechToText:
    # "small" is ~2x more accurate but measured ~2.8 s per utterance on this
    # CPU (vs ~0.7 s for "base") — it blows the 2 s budget. Stay on base.
    def __init__(self, model_size: str = "base") -> None:
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
