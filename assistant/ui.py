"""The assistant's window: animated face, chat log, typed chat, hold-to-talk.

Threads:
- Tk's main loop owns every widget; other threads only post events to a
  queue that the main loop polls.
- The worker thread handles one request at a time (typed text or recorded
  speech): transcribe, then Assistant.respond.
- The mic thread reads the microphone. In push-to-talk mode (the default)
  audio is kept only while Space or the mic button is held, so background
  talk and the assistant's own voice are never transcribed. Hands-free mode
  listens with voice detection whenever the assistant is quiet.
- The camera thread (only while "Hands" is on) tracks one hand and posts
  gesture events; see handmenu.py.
"""

import queue
import threading
import time
import tkinter as tk
from collections import deque
from typing import Callable

import numpy as np

from .audio_io import CHUNK_SAMPLES, SAMPLE_RATE, Microphone
from .face import FACES
from .handmenu import HandControl, Overlay
from .hotkey import TalkKey
from .orb import Orb
from .router import wants_to_quit

Log = Callable[[str, str], None]  # (kind, text); kind is "you", "bot" or "note"

NAME = "JARVIS"
GREETING = "Hello Aditya, what are we working on today?"
CHUNK_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE
MIN_RECORDING_MS = 300  # shorter presses are taps, not speech
MAX_RECORDING_MS = 30_000
RELEASE_TAIL_MS = 200  # keep recording briefly after release: the last syllable trails
PRE_ROLL_MS = 320  # and keep this much from before the press: the first one starts early
ECHO_TAIL_S = 0.35  # hands-free: wait this long after we stop talking before listening
POLL_MS = 30

BG = "#070b12"
PANEL = "#0b1220"
FIELD = "#0f172a"
TEXT = "#e2e8f0"
MUTED = "#64748b"
YOU = "#93c5fd"
ACCENT = "#22d3ee"
DANGER = "#f87171"


class Session:
    """The assistant's threads; talks to the window only through `events`."""

    def __init__(self, assistant, stt, detector, events: queue.Queue, log: Log) -> None:
        self.assistant = assistant
        self._stt = stt
        self._detector = detector
        self._events = events
        self._log = log
        self._requests: queue.Queue = queue.Queue()
        self._closing = threading.Event()
        self._held = threading.Event()
        self.hands_free = False
        self.busy = False  # worker is transcribing or answering
        self.recording = False  # push-to-talk audio is being kept
        self.mic_ok = stt is not None
        self.mic_level = 0.0
        self._recent: deque[np.ndarray] = deque(maxlen=PRE_ROLL_MS // CHUNK_MS)
        self._use_pre_roll = False
        self._drop_recording = False
        self._recent_at = 0.0  # when the newest pre-roll chunk arrived
        self._owner = ""  # which input started the current recording
        self._quiet_since = 0.0
        # Bumped by every Stop; a request that was already in progress when
        # Stop came (e.g. still being transcribed) is then not answered.
        self._interrupts = 0
        threading.Thread(target=self._work, daemon=True, name="worker").start()
        if stt is not None:
            threading.Thread(target=self._listen, daemon=True, name="mic").start()

    # ----- called from the window --------------------------------------

    def submit_text(self, text: str) -> None:
        self._requests.put(("text", text, time.perf_counter()))

    def say(self, text: str) -> None:
        self._requests.put(("say", text, time.perf_counter()))

    def press_to_talk(self, source: str = "key") -> None:
        """Start recording. `source` (space, right ctrl, orb, button) owns it:
        only the same source's release ends it."""
        if self._held.is_set() or not self.mic_ok:
            return  # key auto-repeat, another source already talking, or no microphone
        # The moment before the press holds the start of the first word,
        # unless the assistant was talking: then it's our own voice. The kept
        # audio reaches back PRE_ROLL_MS, so all of it must be after the echo
        # died, and it must be fresh (hands-free listening doesn't refill it).
        now = time.monotonic()
        self._use_pre_roll = (
            not (self.assistant.player.busy or self.busy)
            and now - self._quiet_since >= ECHO_TAIL_S + PRE_ROLL_MS / 1000
            and now - self._recent_at < 0.15
        )
        self.interrupt()  # talking over the assistant stops it
        self._drop_recording = False
        self._owner = source
        self._held.set()

    def release_to_talk(self, source: str | None = None) -> None:
        """End the recording. None (focus lost, closing) ends it whoever started it."""
        if source is None or source == self._owner:
            self._held.clear()

    def cancel_talk(self, source: str | None = None) -> None:
        """It was a keyboard shortcut, not speech: drop what was recorded."""
        if source is None or source == self._owner:
            self._drop_recording = True
            self._held.clear()

    def interrupt(self) -> None:
        self._interrupts += 1
        self.assistant.interrupt()

    def close(self) -> None:
        self._closing.set()
        self._held.clear()
        self.interrupt()
        self._requests.put(("close", "", 0.0))

    @property
    def listening(self) -> bool:
        return self.recording or (self.hands_free and self._detector.in_speech)

    # ----- worker thread -----------------------------------------------

    def _post(self, *event) -> None:
        self._events.put(event)

    def _work(self) -> None:
        while True:
            kind, payload, started_at = self._requests.get()
            if kind == "close" or self._closing.is_set():
                return
            self.busy = True
            interrupts = self._interrupts
            try:
                if kind == "say":
                    self.assistant.say(payload)
                    continue
                text = payload
                if kind == "voice":
                    text = self._stt.transcribe(payload)
                    ms = (time.perf_counter() - started_at) * 1000
                    if not text:
                        self._log("note", "(heard nothing I could make out)")
                        continue
                    self._log("note", f"[speech to text {ms:.0f} ms]")
                self._log("you", text)
                if wants_to_quit(text):
                    self._closing.set()  # nothing queued after this runs
                    self.assistant.say("Goodbye.")
                    self.assistant.player.wait_until_done(timeout=3)
                    self._post("quit")
                    return
                if self._interrupts != interrupts or self._closing.is_set():
                    continue  # Stop was pressed while we were transcribing
                self.assistant.respond(text, started_at)
            except Exception as exc:  # noqa: BLE001 — keep the window alive
                self._log("note", f"(something went wrong: {type(exc).__name__}: {exc})")
            finally:
                self.busy = False

    # ----- mic thread --------------------------------------------------

    def _assistant_quiet(self) -> bool:
        """True once we've been silent long enough that the room is quiet too."""
        if self.busy or self.assistant.player.busy or not self._requests.empty():
            self._quiet_since = time.monotonic()
            return False
        return time.monotonic() - self._quiet_since >= ECHO_TAIL_S

    def _listen(self) -> None:
        try:
            with Microphone() as mic:
                while not self._closing.is_set():
                    if self._held.is_set():
                        self._record_while_held(mic)
                    elif self.hands_free and self._assistant_quiet():
                        self._recent.clear()  # not refilled while listening
                        audio = self._detector.listen(mic, should_stop=self._stop_hands_free)
                        if audio.size and not self._held.is_set():
                            self._requests.put(("voice", audio, time.perf_counter()))
                    else:
                        self._assistant_quiet()  # track when we went quiet
                        chunk = mic.read_chunk(timeout=0.1)  # not for us, but kept briefly:
                        if chunk is not None:
                            self._recent.append(chunk)  # the pre-roll of a press
                            self._recent_at = time.monotonic()
                        self.mic_level = _level(chunk)
        except Exception as exc:  # noqa: BLE001 — e.g. no microphone
            self._log("note", f"(microphone unavailable: {exc}; you can still type)")
        finally:
            self.mic_ok = False
            self._held.clear()
            self._post("mic_gone")

    def _stop_hands_free(self) -> bool:
        # Also stop when we start talking: the mic would hear our own reply.
        return (
            self._closing.is_set()
            or self._held.is_set()
            or not self.hands_free
            or not self._requests.empty()
            or self.busy
            or self.assistant.player.busy
        )

    def _record_while_held(self, mic: Microphone) -> None:
        chunks: list[np.ndarray] = list(self._recent) if self._use_pre_roll else []
        self._recent.clear()
        self.recording = True
        try:
            tail = RELEASE_TAIL_MS // CHUNK_MS
            while len(chunks) * CHUNK_MS < MAX_RECORDING_MS and not self._closing.is_set():
                if not self._held.is_set():
                    tail -= 1
                    if tail < 0:
                        break
                chunk = mic.read_chunk(timeout=0.1)
                if chunk is not None:
                    chunks.append(chunk)
                    self.mic_level = _level(chunk)
        finally:
            self.recording = False
            self._held.clear()
        if len(chunks) * CHUNK_MS >= MIN_RECORDING_MS and not self._drop_recording:
            self._requests.put(("voice", np.concatenate(chunks), time.perf_counter()))


def _sharp_on_high_dpi() -> None:
    """Tell Windows we scale ourselves; otherwise it stretches the window into a blur."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass  # not Windows, or an old one


def _level(chunk: np.ndarray | None) -> float:
    if chunk is None or not chunk.size:
        return 0.0
    rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / 32768
    return min(1.0, rms * 12)


class Window:
    def __init__(self, load: Callable[[Log], tuple], echo: Log, help_text: str = "") -> None:
        """`load(log)` builds (assistant, stt, detector) in the background.

        Every log line goes to the chat panel and to `echo` (the terminal).
        """
        self._echo = echo
        self._help_text = help_text
        _sharp_on_high_dpi()
        self.root = tk.Tk()
        self.root.title(f"{NAME.title()} — offline assistant")
        self.root.configure(bg=BG)
        scale = self.root.winfo_fpixels("1i") / 96  # 1.0 at 100 % display scaling
        width = min(int(1000 * scale), int(self.root.winfo_screenwidth() * 0.9))
        height = min(int(760 * scale), int(self.root.winfo_screenheight() * 0.85))
        self.root.geometry(f"{width}x{height}+40+20")
        self.root.minsize(int(560 * scale), int(460 * scale))
        self._events: queue.Queue = queue.Queue()
        self.session: Session | None = None
        self.talk_key: TalkKey | None = None
        self.hand_control: HandControl | None = None
        self._button_lit = False
        self._last_kind = ""  # kind of the last chat line, to join a reply's sentences
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        threading.Thread(target=self._load, args=(load,), daemon=True, name="loader").start()

    # ----- layout --------------------------------------------------------

    def _build(self) -> None:
        root = self.root
        # The panel is packed first so that a small window shrinks the face,
        # never the chat box and buttons.
        panel = tk.Frame(root, bg=PANEL)
        panel.pack(fill="x", side="bottom")
        self._faces = [make(root, NAME) for make in FACES.values()]
        for face in self._faces:
            face.canvas.bind("<Button-1>", lambda e, c=face.canvas: c.focus_set())
        self.face = self._faces[0]
        self.face.canvas.pack(fill="both", expand=True)
        self.orb = Orb(
            root, lambda: self._talk_down("orb"), lambda: self._talk_up("orb"), self._show_window,
            menu=[("Show window", self._show_window), ("Hands-free on / off", self._flip_hands_free),
                  ("Camera (hand control) on / off", self._flip_camera), ("Quit", self._quit)],
        )

        log_frame = tk.Frame(panel, bg=PANEL)
        log_frame.pack(fill="x", padx=14, pady=(12, 6))
        self.log = tk.Text(
            log_frame, height=7, bg=PANEL, fg=TEXT, relief="flat", wrap="word",
            font=("Segoe UI", 11), padx=6, pady=4, highlightthickness=0, cursor="arrow",
        )
        scroll = tk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set, state="disabled")
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="x", expand=True)
        self.log.tag_configure("you", foreground=YOU, font=("Segoe UI", 11, "bold"))
        self.log.tag_configure("bot", foreground=TEXT)
        self.log.tag_configure("note", foreground=MUTED, font=("Segoe UI", 9))
        self.log.tag_configure("who_bot", foreground=ACCENT, font=("Segoe UI", 11, "bold"))

        row = tk.Frame(panel, bg=PANEL)
        row.pack(fill="x", padx=14, pady=(0, 6), side="bottom", before=log_frame)
        self.entry = tk.Entry(
            row, bg=FIELD, fg=TEXT, insertbackground=TEXT, relief="flat",
            font=("Segoe UI", 12), highlightthickness=1, highlightcolor=ACCENT,
            highlightbackground="#1e293b",
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.entry.bind("<Return>", self._send)
        # Widget bindings run before the Entry inserts the space, so an empty
        # chat box can still be push-to-talk.
        self.entry.bind("<KeyPress-space>", self._space_down)
        self.entry.bind("<KeyRelease-space>", self._space_up)
        self._button(row, "Send", self._send).pack(side="left", padx=(8, 0))
        self.mic_button = self._button(row, "🎤 Hold to talk", None)
        self.mic_button.pack(side="left", padx=(8, 0))
        # Tk sends the release to the button even if the mouse left it.
        self.mic_button.bind("<ButtonPress-1>", lambda e: self._talk_down("button"))
        self.mic_button.bind("<ButtonRelease-1>", lambda e: self._talk_up("button"))
        self._button(row, "■ Stop", self._stop_speaking, fg=DANGER).pack(side="left", padx=(8, 0))
        self.hands_free = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row, text="Hands-free", variable=self.hands_free, command=self._toggle_hands_free,
            bg=PANEL, fg=TEXT, selectcolor=FIELD, activebackground=PANEL, activeforeground=TEXT,
            font=("Segoe UI", 10), takefocus=0,
        ).pack(side="left", padx=(10, 0))
        self.hands_on = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row, text="✋ Hands", variable=self.hands_on, command=self._toggle_camera,
            bg=PANEL, fg=TEXT, selectcolor=FIELD, activebackground=PANEL, activeforeground=TEXT,
            font=("Segoe UI", 10), takefocus=0,
        ).pack(side="left", padx=(6, 0))

        self.status = tk.Label(
            panel, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), anchor="w",
            text="Loading models…",
        )
        self.status.pack(fill="x", padx=16, pady=(0, 10), side="bottom", before=row)

        root.bind("<KeyPress-space>", self._space_down)
        root.bind("<KeyRelease-space>", self._space_up)
        root.bind("<Escape>", lambda e: self._stop_speaking())
        root.bind("<F11>", lambda e: self._fullscreen(not root.attributes("-fullscreen")))
        # A Space release we never see (window switched while holding it)
        # must not leave the mic recording.
        root.bind("<FocusOut>", lambda e: root.after(50, self._release_if_unfocused))

    def _release_if_unfocused(self) -> None:
        if self.root.focus_get() is None:
            self._talk_up("space")

    def _button(self, parent, text, command, fg=TEXT) -> tk.Button:
        # takefocus=0: Space must never "click" a focused button.
        return tk.Button(
            parent, text=text, command=command, bg=FIELD, fg=fg, activebackground="#1e293b",
            activeforeground=fg, relief="flat", font=("Segoe UI", 10), padx=12, pady=6, cursor="hand2",
            takefocus=0,
        )

    # ----- startup ---------------------------------------------------------

    def _load(self, load) -> None:
        try:
            assistant, stt, detector = load(self._log)
        except Exception as exc:  # noqa: BLE001 — show it instead of dying silently
            self._events.put(("failed", str(exc)))
            return
        self._events.put(("ready", assistant, stt, detector))

    def _log(self, kind: str, text: str) -> None:
        """Safe from any thread: the chat panel is updated by the main loop."""
        if kind == "you":
            self._echo("note", f"You: {text}")
        else:
            self._echo(kind, text)
        self._events.put(("log", kind, text))

    # ----- input -------------------------------------------------------------

    def _typing(self) -> bool:
        """Space types into the chat box once it has text; otherwise it's push-to-talk."""
        return self.root.focus_get() is self.entry and bool(self.entry.get())

    def _space_down(self, event) -> str | None:
        if self._typing():
            return None
        self._talk_down("space")
        return "break"

    def _space_up(self, event) -> str | None:
        if self.session is None or not self.session.recording and self._typing():
            return None
        self._talk_up("space")
        return "break"

    def _talk_down(self, source: str) -> None:
        if self.session is not None:
            self.session.press_to_talk(source)

    def _talk_up(self, source: str) -> None:
        if self.session is not None:
            self.session.release_to_talk(source)

    def _send(self, event=None) -> None:
        text = self.entry.get().strip()
        if self.session is None:
            return
        self.entry.delete(0, "end")
        if text:
            self.session.submit_text(text)

    def _stop_speaking(self) -> None:
        if self.session is not None:
            self.session.interrupt()

    def _toggle_hands_free(self) -> None:
        if self.session is not None:
            self.session.hands_free = self.hands_free.get()
        self._show_hint()

    def _flip_hands_free(self) -> None:
        self.hands_free.set(not self.hands_free.get())
        self._toggle_hands_free()

    def _toggle_camera(self) -> str | None:
        """Follow the checkbox; returns why the camera couldn't start, if it couldn't."""
        if self.hand_control is None:
            self.hands_on.set(False)
            return "Hand control isn't ready yet."
        if self.hands_on.get():
            return self.hand_control.start()
        self.hand_control.stop()
        return None

    def _flip_camera(self) -> None:
        self.hands_on.set(not self.hands_on.get())
        self._toggle_camera()

    def _camera_state(self, on: bool) -> None:
        self.hands_on.set(on)
        self.orb.set_camera(on)

    def _show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _fullscreen(self, on: bool) -> None:
        self.root.attributes("-fullscreen", on)

    def _next_face(self) -> None:
        self.face.canvas.pack_forget()
        self.face = self._faces[(self._faces.index(self.face) + 1) % len(self._faces)]
        self.face.canvas.pack(fill="both", expand=True)

    def _ui_actions(self) -> dict[str, Callable[[], str]]:
        """Window controls for voice commands. They run on the worker thread,
        so each only posts to the main loop and returns what to say."""

        def post(action: Callable[[], None], reply: str) -> Callable[[], str]:
            def run() -> str:
                self._events.put(("call", action))
                return reply

            return run

        def set_hands_free(on: bool) -> None:
            self.hands_free.set(on)
            self._toggle_hands_free()

        def camera(on: bool) -> Callable[[], str]:
            def run() -> str:
                # Wait for the main loop to try, so the reply says what happened.
                done, outcome = threading.Event(), {}

                def act() -> None:
                    self.hands_on.set(on)
                    outcome["error"] = self._toggle_camera()
                    done.set()

                self._events.put(("call", act))
                if not on:
                    return "Camera off."
                if not done.wait(3) or outcome.get("error"):
                    return "I couldn't turn the camera on. The chat says why."
                return "Camera on. Hold up your palm for the menu."

            return run

        return {
            "hands_free_on": post(lambda: set_hands_free(True), "Hands-free is on. Just talk."),
            "hands_free_off": post(lambda: set_hands_free(False), "Back to push to talk."),
            "camera_on": camera(True),
            "camera_off": camera(False),
            "fullscreen_on": post(lambda: self._fullscreen(True), "Full screen."),
            "fullscreen_off": post(lambda: self._fullscreen(False), "Okay."),
            "next_face": post(self._next_face, "New look."),
            "hide": post(self.root.iconify, "I'll be up in the corner."),
            "show": post(self._show_window, "Here I am."),
        }

    # ----- event loop --------------------------------------------------------

    def _poll(self) -> None:
        try:
            while True:
                self._handle(self._events.get_nowait())
        except queue.Empty:
            pass
        except Exception as exc:  # noqa: BLE001 — a bad event must not freeze the window
            self._echo("note", f"(window trouble: {type(exc).__name__}: {exc})")
        finally:
            try:
                self._update_face()
            finally:
                self.root.after(POLL_MS, self._poll)

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "log":
            self._append(event[1], event[2])
        elif kind == "call":
            event[1]()
        elif kind == "ready":
            _, assistant, stt, detector = event
            self.session = Session(assistant, stt, detector, self._events, self._log)
            assistant.ui_actions = self._ui_actions()
            self.hand_control = HandControl(
                post=lambda fn: self._events.put(("call", fn)),
                log=self._log,
                workspace=lambda: assistant.workspace,
                on_state=self._camera_state,
                overlay=lambda: Overlay(self.root),
            )
            if stt is None:
                self.mic_button.configure(state="disabled", text="🎤 (no mic)")
            else:
                session = self.session
                self.talk_key = TalkKey(
                    lambda: session.press_to_talk("right ctrl"),
                    lambda: session.release_to_talk("right ctrl"),
                    lambda: session.cancel_talk("right ctrl"),
                    log=lambda text: self._log("note", text),
                )
            self._show_hint()
            self.entry.focus_set()
            self.session.say(GREETING)
        elif kind == "failed":
            self._append("note", f"Could not start: {event[1]}")
            self._append("note", self._help_text)
            self.status.configure(text="Not running.")
        elif kind == "mic_gone":
            self.mic_button.configure(state="disabled", text="🎤 (no mic)")
        elif kind == "quit":
            self.entry.configure(state="disabled")
            self.root.after(200, self.root.destroy)

    def _update_face(self) -> None:
        s = self.session
        player = s.assistant.player if s else None
        if s is None:
            state, level = "loading", 0.0
        elif s.listening or s.recording:
            state, level = "listening", s.mic_level
        elif player.busy:
            state, level = "speaking", min(1.0, player.level * 8)
        elif s.busy:
            state, level = "thinking", 0.0
        else:
            state, level = "idle", 0.0
        self.face.set_state(state)
        self.face.set_level(level)
        self.orb.set_state(state, level)
        recording = bool(s and s.recording)
        if recording != self._button_lit:  # reconfigure only on change
            self._button_lit = recording
            self.mic_button.configure(bg="#14532d" if recording else FIELD)

    def _show_hint(self) -> None:
        if self.hands_free.get():
            hint = "Hands-free: just talk. Esc or ■ Stop cuts a reply short. Say or type “stop” to quit."
        else:
            hint = ("Hold Right Ctrl (any window), Space, or the orb to talk. Esc or ■ Stop cuts a "
                    "reply short. F11 full screen. Say “what can you do” for help, “stop” to quit.")
        self.status.configure(text=hint)

    def _append(self, kind: str, text: str) -> None:
        self.log.configure(state="normal")
        if kind == "you":
            self.log.insert("end", "You  ", "you")
            self.log.insert("end", text + "\n", "you")
        elif kind == "bot" and self._last_kind == "bot":
            # The next sentence of the same reply: continue the paragraph.
            self.log.delete("end-2c")  # the paragraph's newline
            self.log.insert("end", " " + text + "\n", "bot")
        elif kind == "bot":
            self.log.insert("end", f"{NAME.title()}  ", "who_bot")
            self.log.insert("end", text + "\n", "bot")
        else:
            self.log.insert("end", text + "\n", "note")
        self._last_kind = kind
        self.log.configure(state="disabled")
        self.log.see("end")

    def _quit(self) -> None:
        if self.hand_control is not None:
            self.hand_control.stop()
        if self.session is not None:
            self.session.close()
        self.root.destroy()

    def run(self) -> None:
        for face in self._faces:
            face.start()
        self._poll()
        try:
            self.root.mainloop()
        finally:
            if self.talk_key is not None:
                self.talk_key.stop()
            if self.hand_control is not None:
                self.hand_control.stop()
            if self.session is not None:
                self.session.close()
                self.session.assistant.player.close()
