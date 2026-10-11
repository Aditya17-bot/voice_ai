"""The assistant's face: a procedural circuit board drawn on a Tk canvas.

Signals run along the traces from a central chip. The board shows what the
assistant is doing:
- idle: slow, dim pulses
- listening: pulses flow inward to the chip, green
- thinking: fast amber pulses, the chip's ring spins
- speaking: bright pulses outward, the chip glows with the voice

Inspired by the "living circuit board" idea in jaredrhod/ai-visualizer; the
code here is our own, plain tkinter, no dependencies.
"""

import math
import random
import tkinter as tk

FRAME_MS = 33  # ~30 fps
BACKGROUND = "#05080d"
TRACE = "#0d2a33"
PAD = "#123844"
CHIP_FILL = "#0a1219"
CHIP_EDGE = "#1f4b57"
LABEL = "#cfe9f0"

# state -> (pulse color, pulses per second, speed px/s, inward?)
STATES = {
    "loading": ("#475569", 3, 90, False),
    "idle": ("#22d3ee", 4, 110, False),
    "listening": ("#34d399", 14, 210, True),
    "thinking": ("#fbbf24", 22, 340, False),
    "speaking": ("#67e8f9", 10, 260, False),
}
STATE_TEXT = {
    "loading": "WAKING UP",
    "idle": "READY",
    "listening": "LISTENING",
    "thinking": "THINKING",
    "speaking": "SPEAKING",
}
TRACES = 56
MAX_PULSES = 90


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colors; t=0 gives a, t=1 gives b."""
    t = max(0.0, min(1.0, t))
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return f"#{round(ra + (rb - ra) * t):02x}{round(ga + (gb - ga) * t):02x}{round(ba + (bb - ba) * t):02x}"


class _Trace:
    """A polyline from a chip pin to a pad near the edge."""

    def __init__(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self.lengths = [0.0]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            self.lengths.append(self.lengths[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = self.lengths[-1]
        self.heat = 0.0  # pad flash, 0..1
        self.pad_item = 0

    def at(self, distance: float) -> tuple[float, float]:
        distance = max(0.0, min(self.length, distance))
        for i in range(1, len(self.points)):
            if distance <= self.lengths[i]:
                (x0, y0), (x1, y1) = self.points[i - 1], self.points[i]
                span = self.lengths[i] - self.lengths[i - 1] or 1.0
                f = (distance - self.lengths[i - 1]) / span
                return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        return self.points[-1]


class Face:
    def __init__(self, parent: tk.Widget, name: str = "JARVIS") -> None:
        self.canvas = tk.Canvas(parent, bg=BACKGROUND, highlightthickness=0)
        self._name = name
        self._state = "loading"
        self._level = 0.0  # smoothed voice/mic loudness, 0..1
        self._target_level = 0.0
        self._traces: list[_Trace] = []
        self._pulses: list[dict] = []
        self._spawn_debt = 0.0
        self._spin = 0.0
        self._size = (0, 0)
        self._s = 1.0  # display scaling, set when the board is built
        self._rebuild: str | None = None  # pending after() id for a resize
        self._rng = random.Random(7)
        self.canvas.bind("<Configure>", self._on_resize)

    # ----- public -------------------------------------------------------

    def set_state(self, state: str) -> None:
        if state in STATES and state != self._state:
            self._state = state
            self.canvas.itemconfigure("status", text=STATE_TEXT[state])

    def set_level(self, level: float) -> None:
        """Loudness of the voice (speaking) or the mic (listening), 0..1."""
        self._target_level = max(0.0, min(1.0, level))

    def start(self) -> None:
        self._tick()

    # ----- building the board --------------------------------------------

    def _on_resize(self, event: tk.Event) -> None:
        size = (event.width, event.height)
        if size != self._size and event.width > 50 and event.height > 50:
            self._size = size
            # Dragging the window edge sends many of these; rebuild once it settles.
            if self._rebuild is not None:
                self.canvas.after_cancel(self._rebuild)
            self._rebuild = self.canvas.after(120, self._build)

    def _build(self) -> None:
        self._rebuild = None
        c = self.canvas
        c.delete("all")
        self._pulses.clear()
        w, h = self._size
        s = self._s = c.winfo_fpixels("1i") / 96  # pixel sizes below are at 100 % scaling
        cx, cy = w / 2, h / 2
        chip_w, chip_h = min(260 * s, w * 0.34), min(130 * s, h * 0.3)
        self._chip = (cx - chip_w / 2, cy - chip_h / 2, cx + chip_w / 2, cy + chip_h / 2)
        rng = random.Random(7)  # same board every time for a given size
        self._traces = [self._make_trace(rng, i) for i in range(TRACES)]
        for trace in self._traces:
            c.create_line(*[v for p in trace.points for v in p], fill=TRACE, width=2 * s, capstyle="round")
            x, y = trace.points[-1]
            r = 4 * s
            trace.pad_item = c.create_oval(x - r, y - r, x + r, y + r, fill=PAD, outline="")
        self._draw_chip()

    def _make_trace(self, rng: random.Random, index: int) -> _Trace:
        """Leave the chip from a pin, run straight, bend 45 degrees, run to near the edge."""
        x0, y0, x1, y1 = self._chip
        w, h = self._size
        side = index % 4
        f = (index // 4 + 0.5) / (TRACES / 4)  # spread pins evenly along each side
        if side == 0:  # top
            start, out = (x0 + (x1 - x0) * f, y0), (0, -1)
        elif side == 1:  # bottom
            start, out = (x0 + (x1 - x0) * f, y1), (0, 1)
        elif side == 2:  # left
            start, out = (x0, y0 + (y1 - y0) * f), (-1, 0)
        else:  # right
            start, out = (x1, y0 + (y1 - y0) * f), (1, 0)
        points = [start]
        x, y = start
        reach = (h if out[1] else w) / 2
        run = reach * rng.uniform(0.08, 0.3)
        x, y = x + out[0] * run, y + out[1] * run
        points.append((x, y))
        # Bend diagonally away from the chip's middle, then straight out again.
        lateral = (f - 0.5) * 2
        bend = reach * rng.uniform(0.1, 0.35)
        dx = out[0] * bend + (out[1] != 0) * math.copysign(bend, lateral) * abs(lateral)
        dy = out[1] * bend + (out[0] != 0) * math.copysign(bend, lateral) * abs(lateral)
        x, y = x + dx, y + dy
        points.append((x, y))
        margin = 18 * self._s
        if out[0]:
            x = margin if out[0] < 0 else w - margin
            x = x if rng.random() < 0.6 else (points[-1][0] + x) / 2
        else:
            y = margin if out[1] < 0 else h - margin
            y = y if rng.random() < 0.6 else (points[-1][1] + y) / 2
        points.append((min(max(x, margin), w - margin), min(max(y, margin), h - margin)))
        return _Trace(points)

    def _draw_chip(self) -> None:
        c = self.canvas
        x0, y0, x1, y1 = self._chip
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = self._s
        c.create_rectangle(x0 - 10 * s, y0 - 10 * s, x1 + 10 * s, y1 + 10 * s, outline="", fill="",
                           tags="glow", width=6 * s)
        c.create_rectangle(x0, y0, x1, y1, fill=CHIP_FILL, outline=CHIP_EDGE, width=2 * s, tags="chip")
        chip_h = y1 - y0
        c.create_text(cx, y0 + chip_h * 0.3, text=self._name, fill=LABEL, font=("Consolas", 22, "bold"))
        c.create_text(cx, y0 + chip_h * 0.52, text=STATE_TEXT[self._state], fill="#5b8794",
                      font=("Consolas", 10), tags="status")
        # Voice meter in the chip's bottom quarter, clear of the text.
        self._bar_base = y1 - chip_h * 0.08
        self._bar_max = chip_h * 0.22
        self._bars = []
        bars = 16
        span = (x1 - x0) * 0.7
        for i in range(bars):
            bx = cx - span / 2 + span * (i + 0.5) / bars
            self._bars.append(c.create_line(bx, y1, bx, y1, fill="#1f4b57", width=3 * s))
        # Spinner arc, shown while thinking.
        r = (y1 - y0) * 0.75
        self._ring = c.create_arc(cx - r * 1.9, cy - r, cx + r * 1.9, cy + r, start=0, extent=0,
                                  style="arc", outline="#fbbf24", width=2 * s)

    # ----- animation -------------------------------------------------------

    def _tick(self) -> None:
        try:
            if self.canvas.winfo_ismapped():  # the other face is showing: rest
                self._frame(FRAME_MS / 1000)
        finally:
            self.canvas.after(FRAME_MS, self._tick)

    def _frame(self, dt: float) -> None:
        if not self._traces:
            return
        c = self.canvas
        s = self._s
        color, rate, speed, inward = STATES[self._state]
        speed *= s
        self._level += (self._target_level - self._level) * 0.35
        level = self._level if self._state in ("speaking", "listening") else 0.0

        # Spawn new pulses.
        self._spawn_debt += dt * rate * (1 + 2.5 * level)
        while self._spawn_debt >= 1 and len(self._pulses) < MAX_PULSES:
            self._spawn_debt -= 1
            trace = self._rng.choice(self._traces)
            size = (2.5 + 2.5 * level + self._rng.random()) * s
            dot = c.create_oval(0, 0, 0, 0, fill=color, outline="")
            tail = c.create_line(0, 0, 0, 0, fill=_mix(BACKGROUND, color, 0.45), width=2 * s)
            self._pulses.append({
                "trace": trace, "d": trace.length if inward else 0.0, "v": speed * self._rng.uniform(0.7, 1.3),
                "inward": inward, "size": size, "dot": dot, "tail": tail,
            })
        self._spawn_debt = min(self._spawn_debt, 3)

        # Move pulses; flash a pad when a pulse arrives at it.
        alive = []
        for p in self._pulses:
            step = p["v"] * dt
            p["d"] += -step if p["inward"] else step
            trace = p["trace"]
            if p["d"] < 0 or p["d"] > trace.length:
                if not p["inward"]:
                    trace.heat = 1.0
                c.delete(p["dot"], p["tail"])
                continue
            x, y = trace.at(p["d"])
            tx, ty = trace.at(p["d"] + (14 if p["inward"] else -14) * s)
            r = p["size"]
            c.coords(p["dot"], x - r, y - r, x + r, y + r)
            c.coords(p["tail"], tx, ty, x, y)
            alive.append(p)
        self._pulses = alive

        for trace in self._traces:
            if trace.heat > 0:
                trace.heat = max(0.0, trace.heat - dt * 2.2)
                c.itemconfigure(trace.pad_item, fill=_mix(PAD, color, trace.heat))

        # Chip: glow with the voice, meter bars, spinner while thinking.
        glow = 0.25 + 0.75 * level if self._state in ("speaking", "listening") else 0.15
        if self._state == "thinking":
            glow = 0.35 + 0.25 * math.sin(self._spin / 15)
        c.itemconfigure("glow", outline=_mix(BACKGROUND, color, glow))
        c.itemconfigure("chip", outline=_mix(CHIP_EDGE, color, glow))
        x0, y0, x1, y1 = self._chip
        for i, bar in enumerate(self._bars):
            wave = 0.5 + 0.5 * math.sin(self._spin / 4 + i * 0.9)
            height = self._bar_max * (0.08 + 0.92 * level * (0.4 + 0.6 * wave))
            bx = c.coords(bar)[0]
            c.coords(bar, bx, self._bar_base, bx, self._bar_base - height)
            c.itemconfigure(bar, fill=_mix(CHIP_EDGE, color, 0.3 + level))
        self._spin += 360 * dt
        if self._state == "thinking":
            c.itemconfigure(self._ring, start=-self._spin % 360, extent=70)
        else:
            c.itemconfigure(self._ring, extent=0)


class RadialFace:
    """A particle orb ringed by bars that jump with the voice.

    Same interface as Face. Idle: slow drift and a sonar ripple. Listening:
    bars lean in, green. Thinking: a radar sweep. Speaking: bars and
    particles burst with the loudness of the voice.
    """

    BARS = 72
    PARTICLES = 70

    def __init__(self, parent: tk.Widget, name: str = "JARVIS") -> None:
        self.canvas = tk.Canvas(parent, bg=BACKGROUND, highlightthickness=0)
        self._name = name
        self._state = "loading"
        self._level = 0.0
        self._target_level = 0.0
        self._t = 0.0
        self._size = (0, 0)
        self._rng = random.Random(3)
        self._rebuild: str | None = None
        self.canvas.bind("<Configure>", self._on_resize)

    def set_state(self, state: str) -> None:
        if state in STATES and state != self._state:
            self._state = state
            self.canvas.itemconfigure("status", text=STATE_TEXT[state])

    def set_level(self, level: float) -> None:
        self._target_level = max(0.0, min(1.0, level))

    def start(self) -> None:
        self._tick()

    def _on_resize(self, event: tk.Event) -> None:
        size = (event.width, event.height)
        if size != self._size and event.width > 50 and event.height > 50:
            self._size = size
            if self._rebuild is not None:
                self.canvas.after_cancel(self._rebuild)
            self._rebuild = self.canvas.after(120, self._build)

    def _build(self) -> None:
        self._rebuild = None
        c = self.canvas
        c.delete("all")
        w, h = self._size
        s = self._s = c.winfo_fpixels("1i") / 96
        self._c = (w / 2, h / 2 - 10 * s)
        self._r = min(w, h) * 0.2
        cx, cy, r = *self._c, self._r
        self._ripple = c.create_oval(cx, cy, cx, cy, outline="", width=2 * s)
        self._sweep = c.create_arc(cx - r, cy - r, cx + r, cy + r, start=0, extent=0, style="pieslice",
                                   fill="", outline="")
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=TRACE, width=2 * s, tags="ring")
        self._bars = [c.create_line(cx, cy, cx, cy, width=3 * s, capstyle="round") for _ in range(self.BARS)]
        rng = random.Random(3)
        self._particles = []
        for _ in range(self.PARTICLES):
            dot = c.create_oval(0, 0, 0, 0, outline="")
            self._particles.append([dot, rng.uniform(0, 2 * math.pi), rng.uniform(0.15, 0.9),
                                    rng.uniform(0.3, 1.2) * rng.choice((-1, 1)), rng.uniform(1.2, 2.6)])
        c.create_text(cx, cy + r * 1.75, text=self._name, fill=LABEL, font=("Consolas", 20, "bold"))
        c.create_text(cx, cy + r * 1.75 + 26 * s, text=STATE_TEXT[self._state], fill="#5b8794",
                      font=("Consolas", 10), tags="status")

    def _tick(self) -> None:
        try:
            if self.canvas.winfo_ismapped() and self._rebuild is None and hasattr(self, "_bars"):
                self._frame(FRAME_MS / 1000)
        finally:
            self.canvas.after(FRAME_MS, self._tick)

    def _frame(self, dt: float) -> None:
        c, s = self.canvas, self._s
        cx, cy, r = *self._c, self._r
        color, _rate, _speed, inward = STATES[self._state]
        self._t += dt
        self._level += (self._target_level - self._level) * 0.35
        level = self._level if self._state in ("speaking", "listening") else 0.0
        t = self._t

        # Bars around the ring.
        for i, bar in enumerate(self._bars):
            a = 2 * math.pi * i / self.BARS
            wave = 0.5 + 0.5 * math.sin(t * 6 + i * 0.7) * math.sin(t * 2.3 + i * 0.31)
            if self._state == "thinking":
                wave = 0.5 + 0.5 * math.sin(i * 0.4 - t * 9)
            length = r * (0.06 + (0.08 if self._state != "idle" else 0.03) * wave + 0.55 * level * wave)
            start = r * (1.08 if not inward else 1.08 + 0.25 * wave * level)
            end = start + length * (-1 if inward else 1)
            c.coords(bar, cx + math.cos(a) * start, cy + math.sin(a) * start,
                     cx + math.cos(a) * end, cy + math.sin(a) * end)
            c.itemconfigure(bar, fill=_mix(CHIP_EDGE, color, 0.35 + 0.65 * max(level, wave * 0.4)))

        # Particles orbiting inside.
        spin = 0.6 + 2.5 * level + (1.5 if self._state == "thinking" else 0)
        for p in self._particles:
            dot, angle, radius, speed, size = p
            p[1] = angle = angle + speed * spin * dt
            burst = 1 + 0.25 * level * math.sin(t * 9 + angle * 3)
            pr = r * radius * burst
            x, y = cx + math.cos(angle) * pr, cy + math.sin(angle) * pr
            d = size * s * (1 + level)
            c.coords(dot, x - d, y - d, x + d, y + d)
            c.itemconfigure(dot, fill=_mix(BACKGROUND, color, 0.35 + 0.5 * radius))

        # Idle sonar ripple, thinking radar sweep.
        if self._state in ("idle", "loading"):
            phase = (t % 2.5) / 2.5
            rr = r * (1.1 + 1.2 * phase)
            c.coords(self._ripple, cx - rr, cy - rr, cx + rr, cy + rr)
            c.itemconfigure(self._ripple, outline=_mix(BACKGROUND, color, 0.5 * (1 - phase)))
        else:
            c.itemconfigure(self._ripple, outline="")
        if self._state == "thinking":
            c.itemconfigure(self._sweep, start=(-t * 220) % 360, extent=40, fill=_mix(BACKGROUND, color, 0.25))
        else:
            c.itemconfigure(self._sweep, extent=0, fill="")
        c.itemconfigure("ring", outline=_mix(TRACE, color, 0.2 + 0.6 * level))


FACES = {"circuit board": Face, "orb": RadialFace}
