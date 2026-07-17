"""Offline smoke test: TTS a phrase, resample, run VAD + STT on it.

Verifies the whole pipeline except the physical mic/speaker.
Run: python -m tests.loopback_test
"""

import sys
import time

import numpy as np


def main() -> None:
    from assistant.stt import SpeechToText
    from assistant.tts import TextToSpeech
    from pysilero_vad import SileroVoiceActivityDetector

    phrase = "Hello, this is a test of the voice assistant."

    print("Loading TTS (downloads voice on first run)...")
    tts = TextToSpeech()
    t0 = time.perf_counter()
    audio, rate = tts.synthesize(phrase)
    tts_ms = (time.perf_counter() - t0) * 1000
    print(f"TTS OK: {audio.size} samples @ {rate} Hz in {tts_ms:.0f} ms")
    assert audio.size > 0, "TTS produced no audio"

    # Naive resample to 16 kHz for VAD/STT (fine for a smoke test).
    idx = (np.arange(int(audio.size * 16_000 / rate)) * rate / 16_000).astype(int)
    audio16 = audio[np.minimum(idx, audio.size - 1)]

    print("Running VAD on synthesized audio...")
    vad = SileroVoiceActivityDetector()
    probs = [
        vad(audio16[i : i + 512].tobytes())
        for i in range(0, audio16.size - 512, 512)
    ]
    peak = max(probs)
    print(f"VAD OK: peak speech probability {peak:.2f}")
    assert peak > 0.5, "VAD did not detect speech in TTS output"

    print("Loading STT (downloads whisper base on first run)...")
    stt = SpeechToText()
    t0 = time.perf_counter()
    text = stt.transcribe(audio16)
    stt_ms = (time.perf_counter() - t0) * 1000
    print(f"STT OK in {stt_ms:.0f} ms: {text!r}")
    assert "test" in text.lower(), f"unexpected transcription: {text!r}"

    print("\nLOOPBACK PASS: TTS -> VAD -> STT all working offline.")


if __name__ == "__main__":
    sys.exit(main())
