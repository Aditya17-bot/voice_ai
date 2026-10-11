"""Phase 5: streaming voice assistant with fallback and barge-in — speak over
the bot and it stops and listens. Works best with headphones; on open speakers
the bot may occasionally hear itself despite the stricter barge-in threshold.

`--text` swaps the mic and STT for typed input (replies are still spoken), for
quick testing without talking."""

import argparse
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


def respond(
    text: str,
    started_at: float,
    brain: Brain,
    tts: TextToSpeech,
    player: Player,
    fallback: FallbackEngine,
) -> None:
    """Stream the reply to `text` into the player, with filler and recovery."""
    first_word_ms = None
    try:
        with fallback.watchdog() as watchdog:
            for sentence in brain.stream_reply(text):
                audio, rate = tts.synthesize(sentence)
                watchdog.cancel()
                if first_word_ms is None:
                    first_word_ms = (time.perf_counter() - started_at) * 1000
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


def run_voice(
    stt: SpeechToText,
    detector: UtteranceDetector,
    brain: Brain,
    tts: TextToSpeech,
    player: Player,
    fallback: FallbackEngine,
) -> None:
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
            respond(text, heard_at, brain, tts, player, fallback)
            # While the reply plays, watch for the user speaking over it.
            carryover = detector.monitor_barge_in(mic, player)
            if carryover:
                print("   (interrupted — listening)")
            else:
                mic.drain()  # don't transcribe our own voice


def run_text(
    brain: Brain, tts: TextToSpeech, player: Player, fallback: FallbackEngine
) -> None:
    print("Ready. Type a message; empty line or Ctrl+C to quit.\n")
    while True:
        try:
            text = input("⌨️  You: ").strip()
        except EOFError:
            return
        if not text:
            return
        respond(text, time.perf_counter(), brain, tts, player, fallback)
        player.wait_until_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline voice assistant")
    parser.add_argument(
        "--text", action="store_true", help="type instead of speaking (no mic/STT)"
    )
    args = parser.parse_args()

    print("Loading models (first run downloads them)...")
    try:
        if not args.text:  # typed input needs no mic models
            stt = SpeechToText()
            detector = UtteranceDetector()
        tts = TextToSpeech()
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

    if args.text:
        run_text(brain, tts, player, fallback)
    else:
        run_voice(stt, detector, brain, tts, player, fallback)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
