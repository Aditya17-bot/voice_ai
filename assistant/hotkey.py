"""Hold Right Ctrl to talk, from any window (Windows only).

No keyboard hook: a thread asks Windows every 20 ms whether Right Ctrl is
down (GetAsyncKeyState). Nothing is recorded or logged. While Right Ctrl is
held, the other keys and mouse buttons are checked too, only to notice that
it's part of a shortcut (Ctrl+C, Ctrl+scroll, Ctrl+click): then the press is
not talk, and anything recorded is dropped.
"""

import sys
import threading
import time
from typing import Callable

POLL_S = 0.02
HOLD_BEFORE_TALK_S = 0.2  # a quick tap or a shortcut never starts recording
VK_RCONTROL = 0xA3
VK_CONTROL = 0x11  # "a Ctrl key is down": also set by Right Ctrl itself
# Every other key and mouse button (0x01-0x06 are the mouse buttons).
_OTHER_KEYS = [k for k in range(0x01, 0xFF) if k not in (VK_RCONTROL, VK_CONTROL)]


class TalkKey:
    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        on_cancel: Callable[[], None],
        key: int = VK_RCONTROL,
        log: Callable[[str], None] = print,
    ) -> None:
        self._on_press, self._on_release, self._on_cancel = on_press, on_release, on_cancel
        self._key = key
        self._log = log
        self._stop = threading.Event()
        self.available = sys.platform == "win32"
        if self.available:
            import ctypes

            self._state = ctypes.windll.user32.GetAsyncKeyState
            threading.Thread(target=self._run, daemon=True, name="talk-key").start()

    def _down(self, key: int) -> bool:
        return bool(self._state(key) & 0x8000)

    def _call(self, callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 — the key must keep working
            self._log(f"(talk key trouble: {type(exc).__name__}: {exc})")

    def _run(self) -> None:
        held_since = None  # when Right Ctrl went down
        talking = False
        shortcut = False
        while not self._stop.wait(POLL_S):
            if self._down(self._key):
                if held_since is None:
                    # Already part of a combination if anything else is down.
                    held_since, shortcut = time.monotonic(), any(self._down(k) for k in _OTHER_KEYS)
                elif not shortcut and any(self._down(k) for k in _OTHER_KEYS):
                    shortcut = True
                    if talking:
                        talking = False
                        self._call(self._on_cancel)
                if not shortcut and not talking and time.monotonic() - held_since >= HOLD_BEFORE_TALK_S:
                    talking = True
                    self._call(self._on_press)
            elif held_since is not None:
                held_since = None
                if talking:
                    talking = False
                    self._call(self._on_release)

    def stop(self) -> None:
        self._stop.set()
