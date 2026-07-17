"""Phase 5: streaming voice assistant with fallback and barge-in — speak over
the bot and it stops and listens. Works best with headphones; on open speakers
the bot may occasionally hear itself despite the stricter barge-in threshold."""

import sys
import time

from .audio_io import Microphone, Player

# Windows consoles often default to cp1252, which can't print the emoji cues.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from .brain import Brain
from .fallback import FallbackEngine
from .stt import SpeechToText
from .tts import TextToSpeech
from .vad import UtteranceDetector


def main() -> None:
    print("Loading models (first run downloads them)...")
    try:
        stt = SpeechToText()
        tts = TextToSpeech()
        detector = UtteranceDetector()
        brain = Brain()
        brain.warm_up()
    except Exception as exc:  # noqa: BLE001 — startup failure needs a clear message
        print(
            f"Could not start: {exc}\n"
            "Check that Ollama is running (`ollama serve`) and that the first "
            "run had internet once to download STT/TTS models."
        )
        return
    player = Player()
    fallback = FallbackEngine(tts, player)  # pre-generates filler clips
    print("Ready. Ctrl+C to quit.\n")

    with Microphone() as mic:
        carryover = []
        while True:
            print("🎤 Listening…")
            utterance = detector.listen(mic, carryover)
            carryover = []
            heard_at = time.perf_counter()
            text = stt.transcribe(utterance)
            stt_ms = (time.perf_counter() - heard_at) * 1000
            if not text:
                print("   (heard nothing intelligible)\n")
                continue
            print(f"   You: {text}  [STT {stt_ms:.0f} ms]")

            first_word_ms = None
            try:
                with fallback.watchdog() as watchdog:
                    for sentence in brain.stream_reply(text):
                        audio, rate = tts.synthesize(sentence)
                        watchdog.cancel()
                        if first_word_ms is None:
                            first_word_ms = (time.perf_counter() - heard_at) * 1000
                        player.enqueue(audio, rate)
                        print(f"   Bot: {sentence}")
            except Exception as exc:  # noqa: BLE001 — keep the conversation alive
                print(f"   (LLM trouble: {type(exc).__name__}: {exc})")
                if first_word_ms is None:
                    # Nothing was said yet — recover conversationally. If the
                    # answer was already partially spoken, saying "repeat that?"
                    # would only confuse; let the partial answer stand.
                    fallback.speak_recovery()
            if first_word_ms is not None:
                print(f"   [first word at {first_word_ms:.0f} ms after you stopped]\n")
            # While the reply plays, watch for the user speaking over it.
            carryover = detector.monitor_barge_in(mic, player)
            if carryover:
                print("   (interrupted — listening)")
            else:
                mic.drain()  # don't transcribe our own voice


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
