"""Voice assistant: streaming replies with fallback and barge-in, plus
read-only desktop actions (open apps and sites, browse and explain folders).
Common commands take the regex fast path; other desktop-sounding requests let
the LLM classify them; everything else is plain conversation.

Speak over the bot and it stops and listens. Works best with headphones; on
open speakers the bot may occasionally hear itself despite the stricter
barge-in threshold.

`--text` swaps the mic and STT for typed input (replies are still spoken), for
quick testing without talking."""

import argparse
import contextlib
import functools
import re
import sys
import time
from pathlib import Path
from typing import Callable

from .audio_io import Microphone, Player

# Windows consoles often default to cp1252, which can't print the emoji cues.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from .brain import Brain
from .classifier import classify_request
from .fallback import FallbackEngine, Watchdog
from .router import (
    Intent,
    Outcome,
    execute,
    might_be_command,
    open_path,
    pick_option,
    refers_back,
    route,
)
from .stt import SpeechToText
from .tools import APP_LABELS, SITES, Ambiguous, ToolError, Workspace
from .tts import TextToSpeech
from .vad import UtteranceDetector

# Pre-synthesized at startup so "open chrome" is answered instantly.
COMMON_CONFIRMATIONS = [
    *(f"Opening {label}." for label in APP_LABELS.values()),
    *(f"Opening {label}." for label, _url in SITES.values()),
]
CACHE_MAX_CHARS = 80  # only short, repeatable phrases are worth keeping as audio
FOLLOWUP_TURNS = 2  # turns after a file explanation that can still ask about it
# Sentence ends followed by a space; "README.md," stays whole.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_DUNDER = re.compile(r"\b__(\w+?)__")
_UNDERSCORE = re.compile(r"(?<=[A-Za-z0-9])_+(?=[A-Za-z0-9])")
# Two word characters before the dot, so "e.g.", "a.m." and "U.S.A." stay
# abbreviations; a letter after it, so "3.5" stays a number.
_EXTENSION_DOT = re.compile(r"(?<=\w\w)\.(?=[A-Za-z][A-Za-z0-9]{0,4}\b)")


def speakable(text: str) -> str:
    """How text should sound: 'voice_input' -> 'voice input', 'main.py' -> 'main dot py'.

    Only what's spoken changes; the printed text keeps the real names.
    """
    text = _DUNDER.sub(r"dunder \1", text)
    return _EXTENSION_DOT.sub(" dot ", _UNDERSCORE.sub(" ", text))


class Assistant:
    """Turns one user utterance into spoken output: a fast-path action, an
    LLM explanation of a file or folder, or a normal streamed reply."""

    def __init__(
        self,
        brain: Brain,
        tts: TextToSpeech,
        player: Player,
        fallback: FallbackEngine,
        workspace: Workspace | None,
    ) -> None:
        self.player = player
        self._brain = brain
        self._tts = tts
        self._fallback = fallback
        self._workspace = workspace
        self._turn = 0
        # Options from the last "did you mean X or Y?", with the request that asked.
        self._pending: tuple[list[Path], str] | None = None
        # Text of the last file or folder explained, for follow-up questions.
        self._followup: str | None = None
        self._followup_turn = 0
        # Short tool confirmations repeat a lot; cache their audio.
        self._synthesize_cached = functools.lru_cache(maxsize=64)(tts.synthesize)
        for phrase in COMMON_CONFIRMATIONS:
            self._synthesize_cached(phrase)

    def respond(self, text: str, started_at: float) -> None:
        self._turn += 1
        if self._workspace is None:
            self._stream(text, started_at)
            return
        pending, self._pending = self._pending, None
        if pending is not None:
            options, asked_for = pending
            choice = pick_option(text, options)
            if choice is not None:
                print(f"   [picked {choice.name}]")
                # Explain with the original request, not "the first one".
                self._perform(lambda: open_path(self._workspace, choice), asked_for, started_at)
                return
        recent = self._turn - self._followup_turn <= FOLLOWUP_TURNS
        if self._followup is not None and recent and refers_back(text):
            # Checked before routing: "explain that again" is about the last
            # file, not a request for a file named "that".
            print("   [follow-up on the last file or folder]")
            self._stream(text, started_at, context=self._followup)
            return
        intent = route(text)
        if intent is not None:
            print(f"   [fast path: {intent.action} {intent.arg!r}]")
            if self._act(intent, text, started_at):
                return
            print("   [not a command after all — asking the LLM]")
        if might_be_command(text):
            self._classify(text, started_at)
        else:
            self._stream(text, started_at)

    def _act(self, intent: Intent, text: str, started_at: float) -> bool:
        """Run an intent and say the result. False: nothing matched, nothing said."""
        return self._perform(lambda: execute(intent, self._workspace), text, started_at)

    def _perform(
        self, action: Callable[[], Outcome | None], text: str, started_at: float
    ) -> bool:
        """Run a tool action and say the result. False: nothing matched, nothing said."""
        try:
            outcome = action()
        except Ambiguous as exc:
            self._followup = None  # asked about something else
            self._pending = (exc.options, text)  # the next answer can pick one
            self._speak(str(exc), started_at)
            return True
        except ToolError as exc:
            self._followup = None
            self._speak(str(exc), started_at)
            return True
        except Exception as exc:  # noqa: BLE001 — e.g. Popen failing; keep the session alive
            self._followup = None
            print(f"   (action failed: {type(exc).__name__}: {exc})")
            self._speak("Sorry, that didn't work. Want to try something else?", started_at)
            return True
        if outcome is None:
            return False
        if outcome.explain:
            self._followup, self._followup_turn = outcome.text, self._turn
            self._stream(text, started_at, context=outcome.text)
        else:
            self._followup = None  # moved on: a new folder, a listing, an app
            self._speak(outcome.text, started_at)
        return True

    def _classify(self, text: str, started_at: float) -> None:
        """Ask the LLM what a request the router missed wants; act, or just reply.

        One filler watchdog covers both LLM calls, so a dropped choice doesn't
        restart the wait for a filler.
        """
        with self._fallback.watchdog() as watchdog:
            try:
                intent = classify_request(self._brain, self._workspace, text)
            except Exception as exc:  # noqa: BLE001 — classifier trouble: just talk
                print(f"   (classifier trouble: {type(exc).__name__}: {exc})")
                intent = None
            print(f"   [LLM picked: {intent or 'nothing, just talk'}]")
            if intent is not None:
                watchdog.cancel()
                if self._act(intent, text, started_at):
                    return
            self._stream(text, started_at, watchdog=watchdog)

    def _speak(self, text: str, started_at: float) -> None:
        """Say fixed text (tool confirmation, listing, or refusal) sentence by sentence."""
        for i, sentence in enumerate(_SENTENCE_BREAK.split(text)):
            spoken = speakable(sentence)
            if len(spoken) <= CACHE_MAX_CHARS:
                audio, rate = self._synthesize_cached(spoken)
            else:
                audio, rate = self._tts.synthesize(spoken)
            self.player.enqueue(audio, rate)
            print(f"   Bot: {sentence}")
            if i == 0:
                elapsed_ms = (time.perf_counter() - started_at) * 1000
        print(f"   [first word at {elapsed_ms:.0f} ms after you stopped]\n")

    def _stream(
        self,
        text: str,
        started_at: float,
        context: str | None = None,
        watchdog: Watchdog | None = None,
    ) -> None:
        """Stream the LLM reply into the player, with filler and recovery.

        Pass a running `watchdog` to keep its filler timing instead of
        starting a new one.
        """
        first_word_ms = None
        try:
            dog = contextlib.nullcontext(watchdog) if watchdog else self._fallback.watchdog()
            with dog as watchdog:
                for sentence in self._brain.stream_reply(text, context=context):
                    audio, rate = self._tts.synthesize(speakable(sentence))
                    watchdog.cancel()
                    if first_word_ms is None:
                        first_word_ms = (time.perf_counter() - started_at) * 1000
                    self.player.enqueue(audio, rate)
                    print(f"   Bot: {sentence}")
        except Exception as exc:  # noqa: BLE001 — keep the conversation alive
            print(f"   (LLM trouble: {type(exc).__name__}: {exc})")
            if first_word_ms is None:
                # Nothing was said yet — recover conversationally. If the
                # answer was already partially spoken, saying "repeat that?"
                # would only confuse; let the partial answer stand.
                self._fallback.speak_recovery()
        if first_word_ms is not None:
            print(f"   [first word at {first_word_ms:.0f} ms after you stopped]\n")


def run_voice(stt: SpeechToText, detector: UtteranceDetector, assistant: Assistant) -> None:
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
            assistant.respond(text, heard_at)
            # While the reply plays, watch for the user speaking over it.
            carryover = detector.monitor_barge_in(mic, assistant.player)
            if carryover:
                print("   (interrupted — listening)")
            else:
                mic.drain()  # don't transcribe our own voice


def run_text(assistant: Assistant) -> None:
    print("Ready. Type a message; empty line or Ctrl+C to quit.\n")
    while True:
        try:
            text = input("⌨️  You: ").strip()
        except EOFError:
            return
        if not text:
            return
        assistant.respond(text, time.perf_counter())
        assistant.player.wait_until_done()


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
    try:
        workspace = Workspace()
    except ValueError as exc:  # no allowed folders — chat still works
        print(f"Desktop actions disabled: {exc}")
        workspace = None
    player = Player()
    fallback = FallbackEngine(tts, player)  # pre-generates filler clips
    assistant = Assistant(brain, tts, player, fallback, workspace)

    if args.text:
        run_text(assistant)
    else:
        run_voice(stt, detector, assistant)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
