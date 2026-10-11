"""Text-to-speech via Piper (local ONNX voice)."""

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "piper"
DEFAULT_VOICE = "en_US-lessac-medium"


def _find_voice_model() -> Path:
    """Locate a Piper .onnx voice in models/piper/, downloading the default if absent."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(MODELS_DIR.glob("*.onnx"))
    if candidates:
        return candidates[0]
    from piper.download_voices import download_voice

    download_voice(DEFAULT_VOICE, MODELS_DIR)
    return MODELS_DIR / f"{DEFAULT_VOICE}.onnx"


class TextToSpeech:
    def __init__(self) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(_find_voice_model()))

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text; returns (int16 audio, sample_rate)."""
        chunks = []
        sample_rate = 22_050
        for chunk in self._voice.synthesize(text):
            sample_rate = chunk.sample_rate
            chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not chunks:
            return np.zeros(0, dtype=np.int16), sample_rate
        return np.concatenate(chunks), sample_rate
