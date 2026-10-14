"""Voice assistant: streaming replies with fallback, plus read-only desktop
actions (open apps and sites, browse and explain folders). Common commands
take the regex fast path; other desktop-sounding requests let the LLM
classify them; everything else is plain conversation.

Modes:
- default: a window with an animated face, chat log, typing, and
  hold-to-talk (see ui.py)
- `--terminal`: the always-listening voice loop in the terminal;
  `--barge-in` lets speech interrupt replies (headphones only: on speakers
  the bot hears itself)
- `--text`: type in the terminal instead of talking (replies still spoken)"""

import argparse
import contextlib
import faulthandler
import functools
import re
import sys
import threading
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
    LIST_ACTIONS,
    is_yes,
    route,
    wants_to_quit,
)
from .memory import Memory, as_said_back

# Handled by the assistant itself: they need no folders and work without them.
SELF_ACTIONS = {
    "ui", "clear_chat", "help", "time", "date", "time_in", "remember", "recall", "forget", "forget_all",
}
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
ECHO_TAIL_S = 0.25  # after playback, how long the room may still echo our voice
PLAYBACK_TIMEOUT_S = 120  # never wait forever on a stuck audio device
# kind is "bot" (a spoken line) or "note" (what happened, timings, errors).
Log = Callable[[str, str], None]


def print_log(kind: str, text: str) -> None:
    print(f"   Bot: {text}" if kind == "bot" else f"   {text}", flush=True)


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
        log: Log = print_log,
        memory: Memory | None = None,
    ) -> None:
        self.player = player
        self._log = log
        self._memory = memory
        brain.memory_note = memory.prompt_note() if memory else ""
        # A yes/no question we asked ("forget everything?"): what "yes" does.
        self._confirm: Callable[[], str] | None = None
        # Set by interrupt(): stop speaking and drop the rest of this reply.
        self._cancel = threading.Event()
        self._watchdog: Watchdog | None = None
        # Window controls a voice command can trigger ("hands free", "full
        # screen"): name -> handler returning what to say. Set by the window.
        self.ui_actions: dict[str, Callable[[], str]] = {}
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

    def interrupt(self) -> None:
        """Stop talking now (Stop button, Esc, or the user starting to talk)."""
        self._cancel.set()
        watchdog = self._watchdog
        if watchdog is not None:
            watchdog.cancel()  # no fillers either
        self.player.stop()

    @property
    def workspace(self) -> Workspace | None:
        """The read-only tools, shared with the hand menu."""
        return self._workspace

    def respond(self, text: str, started_at: float) -> None:
        self._cancel.clear()
        self._turn += 1
        confirm, self._confirm = self._confirm, None
        if confirm is not None and is_yes(text):
            self._speak(confirm(), started_at)  # "yes" to "should I forget …?"
            if self._memory is not None:
                self._brain.memory_note = self._memory.prompt_note()
            return
        if self._workspace is None:
            # No folders allowed: the assistant's own controls still work.
            intent = route(text)
            if intent is not None and intent.action in SELF_ACTIONS and self._act(intent, text, started_at):
                return
            self._stream(text, started_at)
            return
        pending, self._pending = self._pending, None
        if pending is not None:
            options, asked_for = pending
            choice = pick_option(text, options)
            if choice is not None:
                self._log("note", f"[picked {choice.name}]")
                # Explain with the original request, not "the first one".
                self._perform(lambda: open_path(self._workspace, choice), asked_for, started_at)
                return
        recent = self._turn - self._followup_turn <= FOLLOWUP_TURNS
        if self._followup is not None and recent and refers_back(text):
            # Checked before routing: "explain that again" is about the last
            # file, not a request for a file named "that".
            self._log("note", "[follow-up on the last file or folder]")
            self._stream(text, started_at, context=self._followup)
            return
        intent = route(text)
        if intent is not None:
            self._log("note", f"[fast path: {intent.action} {intent.arg!r}]")
            if self._act(intent, text, started_at):
                return
            self._log("note", "[not a command after all — asking the LLM]")
        if might_be_command(text):
            self._classify(text, started_at)
        else:
            self._stream(text, started_at)

    def _act(self, intent: Intent, text: str, started_at: float) -> bool:
        """Run an intent and say the result. False: nothing matched, nothing said."""
        keeps_list = intent.action in LIST_ACTIONS or intent.site == "youtube"
        if self._workspace is not None and not keeps_list:
            self._workspace.forget_lists()  # "the second one" now means something else
        if intent.action in SELF_ACTIONS:
            self._speak(self._own_action(intent), started_at)
            return True
        if self._workspace is None:
            return False
        if intent.action == "play_video":
            # Finding the video takes a few seconds; say so instead of going quiet.
            self._speak(f"Looking for {intent.arg} on YouTube.", started_at)
        return self._perform(lambda: execute(intent, self._workspace), text, started_at)

    def _own_action(self, intent: Intent) -> str:
        """The assistant's own controls and memory; returns what to say."""
        action = intent.action
        if action == "clear_chat":
            self._brain.forget_conversation()
            self._followup = self._pending = None
            return "Okay, fresh start."
        if action == "ui":
            handler = self.ui_actions.get(intent.arg)
            return handler() if handler else "That only works in the window."
        if action in ("help", "time", "date", "time_in"):
            return execute(intent, self._workspace).text
        if self._memory is None:
            return "My memory is switched off."
        if action == "remember":
            said = self._memory.remember(intent.arg)
        elif action == "forget":
            fact = self._memory.find(intent.arg)
            if fact is None:
                return f"I don't remember anything about {as_said_back(intent.arg)}."
            self._confirm = lambda: self._memory.forget(fact)  # done only after a "yes"
            return f"Forget that {as_said_back(fact)}? Say yes to confirm."
        elif action == "forget_all":
            if not self._memory.facts:
                return "My memory is already empty."
            self._confirm = self._memory.forget_all  # done only after a "yes"
            count = len(self._memory.facts)
            return f"Should I forget all {count} things I remember? Say yes to confirm."
        else:  # recall
            return self._memory.recall()
        self._brain.memory_note = self._memory.prompt_note()
        return said

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
            self._log("note", f"(action failed: {type(exc).__name__}: {exc})")
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
            self._watchdog = watchdog
            try:
                intent = classify_request(self._brain, self._workspace, text)
            except Exception as exc:  # noqa: BLE001 — classifier trouble: just talk
                self._log("note", f"(classifier trouble: {type(exc).__name__}: {exc})")
                intent = None
            self._log("note", f"[LLM picked: {intent or 'nothing, just talk'}]")
            if self._cancel.is_set():
                return
            if intent is not None:
                watchdog.cancel()
                if self._act(intent, text, started_at):
                    return
            self._stream(text, started_at, watchdog=watchdog)

    def _enqueue(self, audio, rate: int) -> None:
        self.player.enqueue(audio, rate)
        if self._cancel.is_set():
            self.player.stop()  # Stop landed between our check and the enqueue

    def say(self, text: str) -> None:
        """Speak fixed text outside a turn (greeting, goodbye)."""
        self._cancel.clear()
        self._speak(text, None)

    def _speak(self, text: str, started_at: float | None) -> None:
        """Say fixed text (tool confirmation, listing, or refusal) sentence by sentence."""
        elapsed_ms = None
        for sentence in _SENTENCE_BREAK.split(text):
            if self._cancel.is_set():
                break
            spoken = speakable(sentence)
            if len(spoken) <= CACHE_MAX_CHARS:
                audio, rate = self._synthesize_cached(spoken)
            else:
                audio, rate = self._tts.synthesize(spoken)
            self._enqueue(audio, rate)
            self._log("bot", sentence)
            if elapsed_ms is None and started_at is not None:
                elapsed_ms = (time.perf_counter() - started_at) * 1000
        if elapsed_ms is not None:
            self._log("note", f"[first word at {elapsed_ms:.0f} ms after you stopped]")

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
                self._watchdog = watchdog
                for sentence in self._brain.stream_reply(text, context=context):
                    if self._cancel.is_set():
                        break  # interrupted: drop the rest of the reply
                    audio, rate = self._tts.synthesize(speakable(sentence))
                    watchdog.cancel()
                    if self._cancel.is_set():
                        break
                    if first_word_ms is None:
                        first_word_ms = (time.perf_counter() - started_at) * 1000
                    self._enqueue(audio, rate)
                    self._log("bot", sentence)
        except Exception as exc:  # noqa: BLE001 — keep the conversation alive
            self._log("note", f"(LLM trouble: {type(exc).__name__}: {exc})")
            if first_word_ms is None and not self._cancel.is_set():
                # Nothing was said yet — recover conversationally. If the
                # answer was already partially spoken, saying "repeat that?"
                # would only confuse; let the partial answer stand.
                self._fallback.speak_recovery()
        finally:
            self._watchdog = None
        if first_word_ms is not None:
            self._log("note", f"[first word at {first_word_ms:.0f} ms after you stopped]")


def say_goodbye(assistant: Assistant) -> None:
    assistant.say("Goodbye.")
    assistant.player.wait_until_done(timeout=3)


def run_voice(
    stt: SpeechToText, detector: UtteranceDetector, assistant: Assistant, barge_in: bool
) -> None:
    print('Ready. Say "stop" or "goodbye" (or press Ctrl+C) to quit.\n')

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
            if wants_to_quit(text):
                say_goodbye(assistant)
                return
            assistant.respond(text, heard_at)
            if barge_in:
                # While the reply plays, watch for the user speaking over it.
                # Skip what the mic heard while the reply was being made: it
                # is our own voice and would trigger at once.
                mic.drain()
                carryover = detector.monitor_barge_in(mic, assistant.player)
                if carryover:
                    print("   (interrupted — listening)")
                    continue
            assistant.player.wait_until_done(timeout=PLAYBACK_TIMEOUT_S)
            # Let the room's echo of the last word die, then drop everything
            # the mic heard while we spoke, so we never answer ourselves.
            time.sleep(ECHO_TAIL_S)
            mic.drain()


def run_text(assistant: Assistant) -> None:
    print('Ready. Type a message; "stop", an empty line or Ctrl+C quits.\n')
    while True:
        try:
            text = input("⌨️  You: ").strip()
        except EOFError:
            return
        if not text or wants_to_quit(text):
            say_goodbye(assistant)
            return
        assistant.respond(text, time.perf_counter())
        assistant.player.wait_until_done(timeout=PLAYBACK_TIMEOUT_S)


def load(
    voice: bool, log: Log = print_log
) -> tuple[Assistant, SpeechToText | None, UtteranceDetector | None]:
    """Load the models and wire the assistant. Raises if Ollama is unreachable."""
    stt = SpeechToText() if voice else None  # typed input needs no mic models
    detector = UtteranceDetector() if voice else None
    tts = TextToSpeech()
    brain = Brain()
    brain.warm_up()
    try:
        workspace = Workspace()
    except ValueError as exc:  # no allowed folders — chat still works
        log("note", f"Desktop actions disabled: {exc}")
        workspace = None
    player = Player()
    fallback = FallbackEngine(tts, player)  # pre-generates filler clips
    return Assistant(brain, tts, player, fallback, workspace, log, Memory()), stt, detector


STARTUP_HELP = (
    "Check that Ollama is running (`ollama serve`) and that the first "
    "run had internet once to download STT/TTS models."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline voice assistant")
    parser.add_argument(
        "--text", action="store_true", help="type in the terminal instead of the window (no mic)"
    )
    parser.add_argument(
        "--terminal", action="store_true", help="always-listening voice loop in the terminal, no window"
    )
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="with --terminal: let speech interrupt replies (headphones only)",
    )
    args = parser.parse_args()
    # A crash inside a native library (PortAudio, onnxruntime, CTranslate2)
    # otherwise exits silently; this prints the Python stack that led to it.
    faulthandler.enable()

    if not (args.text or args.terminal):
        from .ui import Window  # tkinter only loads when the window is used

        Window(lambda log: load(voice=True, log=log), echo=print_log, help_text=STARTUP_HELP).run()
        return

    print("Loading models (first run downloads them)...")
    try:
        assistant, stt, detector = load(voice=not args.text)
    except Exception as exc:  # noqa: BLE001 — startup failure needs a clear message
        print(f"Could not start: {exc}\n{STARTUP_HELP}")
        return
    try:
        if args.text:
            run_text(assistant)
        else:
            run_voice(stt, detector, assistant, args.barge_in)
    finally:
        if assistant.player.underruns:
            print(f"   (audio: {assistant.player.underruns} playback underruns — clicks or gaps)")
        assistant.player.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
