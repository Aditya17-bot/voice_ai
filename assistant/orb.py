"""A small always-on-top orb in the screen's top-right corner.

It shows the assistant is running and what it's doing, even with YouTube or
another app in front:
- press and hold: talk (like holding Space)
- click: bring the main window back
- right-click: menu (show window, hands-free, quit)
- Shift + drag: move it
"""

import math
import tkinter as tk
from typing import Callable

SIZE = 64  # at 100 % display scaling
MARGIN = 24
HOLD_TO_TALK_MS = 250  # shorter is a click
TRANSPARENT = "#010203"  # painted where the window should be see-through
FRAME_MS = 40

# state -> (color, label)
LOOK = {
    "loading": ("#475569", "…"),
    "idle": ("#22d3ee", ""),
    "listening": ("#34d399", ""),
    "thinking": ("#fbbf24", ""),
    "speaking": ("#67e8f9", ""),
}


def _mix(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    pa = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


class Orb:
    def __init__(
        self,
        root: tk.Tk,
        on_talk_down: Callable[[], None],
        on_talk_up: Callable[[], None],
        on_show: Callable[[], None],
        menu: list[tuple[str, Callable[[], None]]],
    ) -> None:
        self._talk_down, self._talk_up, self._show = on_talk_down, on_talk_up, on_show
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)  # no title bar or taskbar button
        self.top.attributes("-topmost", True)
        try:
            self.top.attributes("-transparentcolor", TRANSPARENT)  # Windows: round, not square
        except tk.TclError:
            pass
        s = root.winfo_fpixels("1i") / 96
        self._size = size = int(SIZE * s)
        x = root.winfo_screenwidth() - size - int(MARGIN * s)
        self.top.geometry(f"{size}x{size}+{x}+{int(MARGIN * s)}")
        self.canvas = tk.Canvas(self.top, width=size, height=size, bg=TRANSPARENT, highlightthickness=0)
        self.canvas.pack()
        c, pad = self.canvas, size * 0.12
        self._glow = c.create_oval(2, 2, size - 2, size - 2, fill="", outline="", width=max(2, size * 0.06))
        self._body = c.create_oval(pad, pad, size - pad, size - pad, fill="#0a1219", outline="#1f4b57", width=2)
        self._core = c.create_oval(size / 2 - 4, size / 2 - 4, size / 2 + 4, size / 2 + 4, fill="#22d3ee", outline="")
        self._arc = c.create_arc(pad, pad, size - pad, size - pad, start=0, extent=0, style="arc",
                                 outline="#fbbf24", width=max(2, size * 0.05))
        self._state, self._level, self._phase = "loading", 0.0, 0.0
        self._hold_job: str | None = None
        self._talking = False
        self._drag_from: tuple[int, int] | None = None
        self._menu = tk.Menu(self.top, tearoff=0)
        for label, action in menu:
            self._menu.add_command(label=label, command=action)
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<B1-Motion>", self._drag)
        c.bind("<Button-3>", lambda e: self._menu.tk_popup(e.x_root, e.y_root))
        self._tick()

    def set_state(self, state: str, level: float = 0.0) -> None:
        self._state, self._level = state, level

    # ----- mouse ---------------------------------------------------------

    def _press(self, event) -> None:
        if event.state & 0x0001:  # Shift held: move the orb instead
            self._drag_from = (event.x, event.y)
            return
        self._hold_job = self.top.after(HOLD_TO_TALK_MS, self._start_talk)

    def _start_talk(self) -> None:
        self._hold_job = None
        self._talking = True
        self._talk_down()

    def _release(self, event) -> None:
        if self._drag_from is not None:
            self._drag_from = None
        elif self._talking:
            self._talking = False
            self._talk_up()
        elif self._hold_job is not None:
            self.top.after_cancel(self._hold_job)
            self._hold_job = None
            self._show()  # a click

    def _drag(self, event) -> None:
        if self._drag_from is not None:
            dx, dy = event.x - self._drag_from[0], event.y - self._drag_from[1]
            self.top.geometry(f"+{self.top.winfo_x() + dx}+{self.top.winfo_y() + dy}")

    # ----- animation -----------------------------------------------------

    def _tick(self) -> None:
        try:
            if self.top.winfo_ismapped():
                self._frame()
        finally:
            self.top.after(FRAME_MS, self._tick)

    def _frame(self) -> None:
        color, _ = LOOK.get(self._state, LOOK["idle"])
        c, size = self.canvas, self._size
        self._phase += FRAME_MS / 1000
        breathe = 0.5 + 0.5 * math.sin(self._phase * 2.2)
        if self._state == "speaking":
            strength = 0.35 + 0.65 * self._level
        elif self._state == "listening":
            strength = 0.5 + 0.5 * max(self._level, breathe * 0.4)
        elif self._state == "thinking":
            strength = 0.45
        else:
            strength = 0.15 + 0.2 * breathe
        c.itemconfigure(self._glow, outline=_mix("#05080d", color, strength))
        c.itemconfigure(self._body, outline=_mix("#1f4b57", color, strength))
        r = size * (0.08 + 0.1 * strength)
        c.coords(self._core, size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r)
        c.itemconfigure(self._core, fill=_mix("#0a1219", color, 0.4 + 0.6 * strength))
        if self._state == "thinking":
            c.itemconfigure(self._arc, start=(-self._phase * 400) % 360, extent=80)
        else:
            c.itemconfigure(self._arc, extent=0)
