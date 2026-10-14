"""Canvas-drawn widgets for the window: Tk's own buttons and checkboxes are
square and grey, and it has no rounded shapes, so these draw their own.

Every size here is in pixels at 100 % display scaling and is multiplied by
the screen's scale, so the window looks the same on a high-DPI screen.
"""

import tkinter as tk
import tkinter.font as tkfont
from typing import Callable

BG = "#070b12"
PANEL = "#0b1220"
FIELD = "#111a2b"
HOVER = "#1b2740"
BORDER = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#64748b"
YOU = "#93c5fd"
ACCENT = "#22d3ee"
ACCENT_FILL = "#0e7490"
ACCENT_HOVER = "#0891b2"
GO = "#22c55e"
WARN = "#fbbf24"
DANGER = "#f87171"
BOT_BUBBLE = "#131d31"
YOU_BUBBLE = "#1d3557"
FONT = "Segoe UI"


def _scale(widget: tk.Misc) -> float:
    return widget.winfo_fpixels("1i") / 96


def rounded(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float, r: float, **kw) -> int:
    """A rounded rectangle: a smoothed polygon with doubled corner points."""
    r = max(0.0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kw)


class Pill(tk.Canvas):
    """A rounded button. `command` runs on release over the button; leave it
    None to bind press and release yourself (hold to talk). Supports
    configure(text=..., state="disabled" / "normal") like a tk.Button."""

    def __init__(self, parent: tk.Misc, text: str, command: Callable[[], None] | None = None, *,
                 fg: str = TEXT, fill: str = FIELD, hover: str = HOVER, size: int = 10) -> None:
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0, takefocus=0,
                         cursor="hand2")
        self._s = _scale(self)
        self._text, self._command = text, command
        self._fg, self._fill, self._hover = fg, fill, hover
        self._font = tkfont.Font(self, family=FONT, size=size)
        self._enabled = True
        self._inside = self._pressed = False
        self._lit: str | None = None
        self.bind("<Enter>", lambda e: self._set(inside=True))
        self.bind("<Leave>", lambda e: self._set(inside=False))
        self.bind("<ButtonPress-1>", lambda e: self._set(pressed=True), add="+")
        self.bind("<ButtonRelease-1>", self._release, add="+")
        self._draw()

    def configure(self, cnf=None, **kw):  # type: ignore[override]
        if "text" in kw:
            self._text = kw.pop("text")
        if "state" in kw:
            self._enabled = kw.pop("state") != "disabled"
            super().configure(cursor="hand2" if self._enabled else "arrow")
        self._draw()
        if cnf or kw:
            return super().configure(cnf, **kw)
        return None

    config = configure

    def set_lit(self, colour: str | None) -> None:
        """Fill with `colour` (e.g. green while recording), or back to normal."""
        if colour != self._lit:
            self._lit = colour
            self._draw()

    def _set(self, inside: bool | None = None, pressed: bool | None = None) -> None:
        if inside is not None:
            self._inside = inside
        if pressed is not None:
            self._pressed = pressed
        self._draw()

    def _release(self, event) -> None:
        over = 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        self._set(pressed=False)
        if over and self._enabled and self._command is not None:
            self._command()

    def _draw(self) -> None:
        s = self._s
        w = self._font.measure(self._text) + 32 * s
        h = self._font.metrics("linespace") + 16 * s
        super().configure(width=w, height=h)
        self.delete("all")
        if self._lit:
            fill = self._lit
        elif self._enabled and (self._inside or self._pressed):
            fill = self._hover
        else:
            fill = self._fill
        rounded(self, 1, 1, w - 1, h - 1, h / 2, fill=fill, outline=fill)
        self.create_text(w / 2, h / 2, text=self._text, font=self._font,
                         fill=self._fg if self._enabled else MUTED)


class Toggle(tk.Canvas):
    """A switch with a label, bound to a BooleanVar (set from code too)."""

    def __init__(self, parent: tk.Misc, text: str, variable: tk.BooleanVar,
                 command: Callable[[], object] | None = None) -> None:
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0, takefocus=0,
                         cursor="hand2")
        self._s = _scale(self)
        self._text, self._var, self._command = text, variable, command
        self._font = tkfont.Font(self, family=FONT, size=10)
        self.bind("<ButtonRelease-1>", self._flip)
        variable.trace_add("write", lambda *_: self._draw())
        self._draw()

    def _flip(self, event=None) -> None:
        self._var.set(not self._var.get())
        if self._command is not None:
            self._command()

    def _draw(self) -> None:
        s = self._s
        tw, th = 34 * s, 18 * s
        h = max(th, self._font.metrics("linespace")) + 4 * s
        w = tw + 8 * s + self._font.measure(self._text)
        self.configure(width=w, height=h)
        self.delete("all")
        on = bool(self._var.get())
        y = h / 2
        rounded(self, 1, y - th / 2, tw, y + th / 2, th / 2, fill=ACCENT_FILL if on else BORDER,
                outline=ACCENT if on else BORDER)
        r = th / 2 - 3 * s
        cx = tw - th / 2 if on else 1 + th / 2
        self.create_oval(cx - r, y - r, cx + r, y + r, fill=TEXT if on else MUTED, outline="")
        self.create_text(tw + 8 * s, y, text=self._text, anchor="w", font=self._font,
                         fill=TEXT if on else MUTED)


class Chip(tk.Canvas):
    """A small status label with a coloured dot: "● Listening"."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0, takefocus=0)
        self._s = _scale(self)
        self._font = tkfont.Font(self, family=FONT, size=9)
        self._shown: tuple[str, str] | None = None

    def set(self, text: str, colour: str) -> None:
        if (text, colour) == self._shown:  # called 30 times a second: redraw only on change
            return
        self._shown = (text, colour)
        s = self._s
        h = self._font.metrics("linespace") + 10 * s
        w = self._font.measure(text) + 30 * s
        self.configure(width=w, height=h)
        self.delete("all")
        rounded(self, 1, 1, w - 1, h - 1, h / 2, fill=FIELD, outline=BORDER)
        d = 3.5 * s
        self.create_oval(12 * s - d, h / 2 - d, 12 * s + d, h / 2 + d, fill=colour, outline="")
        self.create_text(21 * s, h / 2, text=text, anchor="w", font=self._font, fill=TEXT)


class RoundedEntry(tk.Canvas):
    """A rounded text box with a placeholder. The tk.Entry is `.entry`."""

    def __init__(self, parent: tk.Misc, placeholder: str) -> None:
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self._s = s = _scale(self)
        font = tkfont.Font(self, family=FONT, size=12)
        self._h = font.metrics("linespace") + 20 * s
        self.configure(height=self._h)
        self._text = tk.StringVar(self)
        self.entry = tk.Entry(self, textvariable=self._text, bg=FIELD, fg=TEXT, insertbackground=TEXT,
                              relief="flat", bd=0, highlightthickness=0, font=font)
        self._hint = tk.Label(self.entry, text=placeholder, bg=FIELD, fg=MUTED, font=font, bd=0, padx=0)
        self._hint.bind("<Button-1>", lambda e: self.entry.focus_set())
        self._window = self.create_window(18 * s, self._h / 2, window=self.entry, anchor="w")
        self._focused = False
        self.entry.bind("<FocusIn>", lambda e: self._focus(True), add="+")
        self.entry.bind("<FocusOut>", lambda e: self._focus(False), add="+")
        self._text.trace_add("write", lambda *_: self._placeholder())
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", lambda e: self.entry.focus_set())
        self._placeholder()

    def _focus(self, on: bool) -> None:
        self._focused = on
        self._draw()

    def _placeholder(self) -> None:
        if self._text.get():
            self._hint.place_forget()
        else:
            self._hint.place(x=0, rely=0.5, anchor="w")

    def _draw(self) -> None:
        w, s = self.winfo_width(), self._s
        self.delete("frame")
        edge = ACCENT if self._focused else BORDER
        rounded(self, 1, 1, w - 1, self._h - 1, self._h / 2, fill=FIELD, outline=edge, width=1.5 * s,
                tags="frame")
        self.tag_lower("frame")
        self.itemconfigure(self._window, width=max(10, w - 36 * s))


class Bubble(tk.Canvas):
    """One chat message: wrapped text on a rounded card, at most `max_width` wide."""

    PAD_X, PAD_Y = 14, 9

    def __init__(self, parent: tk.Misc, text: str, fill: str, fg: str, max_width: int) -> None:
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self._s = _scale(self)
        self._font = tkfont.Font(self, family=FONT, size=11)
        self._text, self._fill, self._fg, self._max = text, fill, fg, max_width
        self._draw()

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self._draw()

    def set_max_width(self, max_width: int) -> None:
        if max_width != self._max:
            self._max = max_width
            self._draw()

    def _draw(self) -> None:
        s = self._s
        px, py = self.PAD_X * s, self.PAD_Y * s
        self.delete("all")
        item = self.create_text(px, py, text=self._text, anchor="nw", font=self._font, fill=self._fg,
                                width=max(40, self._max - 2 * px))
        x1, y1, x2, y2 = self.bbox(item)
        w, h = x2 - x1 + 2 * px, y2 - y1 + 2 * py
        self.configure(width=w, height=h)
        card = rounded(self, 1, 1, w - 1, h - 1, 14 * s, fill=self._fill, outline=self._fill)
        self.tag_lower(card)
