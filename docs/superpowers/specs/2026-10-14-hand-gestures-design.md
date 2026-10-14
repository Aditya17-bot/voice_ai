# Hand gesture control (phase 16) — design

Date: 2026-10-14. Status: approved in chat (approach, gestures, modules).

## Goal

Control the desktop with one hand in front of the webcam, Jarvis-style:
open things (YouTube, Claude, a folder, …) from a hand menu, close the
front window, switch windows, switch virtual desktops. Nothing is written,
deleted, moved or run; the read-only rules in CLAUDE.md still hold.

Inspiration: the "barehands" piece of jaredrhod/fullstack-agent (webcam +
Chrome, moves notes around). Idea only — no code borrowed, no browser.

## Decisions (from the user)

- Hand menu (palm opens a ring of tiles, point to hover, pinch to open) —
  not fixed one-gesture-per-target, not voice-only.
- Folder tiles open in File Explorer (Explorer comes back, but only for
  opening the three allowed roots, launched by absolute path).
- MediaPipe hand landmarks + our own gesture rules.
- Gesture set as below; no mouse control, minimize/maximize or tab closing.

## Gestures

Coordinates are MediaPipe's 21 hand points, normalized 0..1, from a
mirrored frame (moving your hand right moves it right on screen).
`scale` = wrist→middle-knuckle distance, so rules work near or far.

| Gesture | Rule | Action |
|---|---|---|
| Open palm, held still 0.6 s | 4 fingers extended, palm centre moves < 0.04 | open hand menu |
| Move hand while menu open | palm centre offset from where the menu opened; its angle picks a tile once > 0.06 away (palm, not fingertip, so pinching doesn't move the pointer) | hover tile |
| "OK" pinch while menu open | thumb tip–index tip < 0.35·scale with at least two of middle/ring/pinky up, 2 frames in a row (a fist never counts as a pinch) | open hovered tile |
| Fist, or hand gone 0.8 s, or 8 s idle | no finger extended | close menu |
| Open hand swipe | palm centre moves > 0.22 horizontally within 0.45 s, vertical < half of that | switch window (left = next, right = previous, like a touchpad) |
| Two-finger (✌) swipe | same motion with index+middle up, ring+pinky folded | switch virtual desktop (left = next/right desktop, right = previous) |
| Thumbs down, held 1.0 s | 4 fingers folded, thumb extended and tip well below the wrist | close the front window (overlay shows "Close <title>?" ring; lowering the hand cancels) |

Guards: every action is followed by a 1 s cooldown; swipes are ignored
while the menu is open; the menu won't open mid-swipe (palm must be still).

## Modules

```
camera thread (hands.py)        main thread (Tk)
 OpenCV grab, ~15 fps      ─►    events from gestures.GestureEngine
 MediaPipe HandLandmarker        ├─ handmenu.Overlay: ring, hover, close ring, toasts
 gestures.GestureEngine          └─ actions: tools.Workspace (tiles), windows.py
```

- `assistant/gestures.py` — pure state machine. `GestureEngine.feed(points | None, t) -> list[Event]`.
  Events: `MenuOpen`, `Hover(i | None)`, `Pick(i)`, `MenuClose`,
  `SwitchWindow(step)`, `SwitchDesktop(step)`, `CloseArming(progress)`,
  `CloseCancel`, `CloseWindow`. No camera, no Windows calls; all thresholds
  are module constants. Unit-tested with synthetic hands.
- `assistant/hands.py` — `HandTracker(on_points, on_stop)`: opens the camera
  (`cv2.CAP_DSHOW`, 640×480), mirrors frames, runs MediaPipe
  `GestureRecognizer` (VIDEO mode, one hand) from
  `models/gesture_recognizer.task`, calls `on_points(points | None, t, label | None)`;
  the label (Open_Palm, Victory, Closed_Fist, Thumb_Down, score ≥ 0.6) decides
  the hand shape, the point rules are the fallback. Drops to ~5 fps after 3 s with no
  hand. Frames live in memory only — never saved or sent. `cv2`/`mediapipe`
  are imported lazily; missing packages or model raise `HandsUnavailable`
  with the fix to run.
- `assistant/windows.py` — `ctypes` `user32`, no new dependency:
  - `foreground()` → (hwnd, title) of the front top-level window.
  - `close_front()` posts `WM_CLOSE` (the app still asks to save). Refuses
    our own process, the desktop (`Progman`, `WorkerW`) and the taskbar.
  - `switch_window(step)` walks the Alt-Tab list (visible, unowned, titled,
    not tool windows, not cloaked i.e. on this desktop, not ours) and
    brings the next/previous one to the front, restoring it if minimized;
    "next" also sends the old front window to the back so repeated swipes
    cycle through all windows.
  - `switch_desktop(step)` sends exactly Win+Ctrl+Left/Right (left Ctrl, so
    the Right Ctrl talk key never sees it). No other keystroke exists.
- `assistant/handmenu.py` — `Overlay`: topmost, click-through, never-focused
  Tk window. Draws the tile ring with the hovered tile lit, the close ring
  with the window title, and short toasts ("→ next window").
- `tools.py`:
  - Claude desktop app in the app allowlist (`claude`, `claude app`,
    `claude desktop`), launched as
    `%WINDIR%\explorer.exe shell:AppsFolder\Claude_pzs8sxrjxfjjc!Claude`.
    Voice "open claude" works too.
  - `Workspace.open_in_explorer(root)` — `%WINDIR%\explorer.exe <root>`
    for one of the allowed root names only, through `_guard`.
- `ui.py` — a "✋ Hands" checkbox; voice "camera on/off", "hand control
  on/off", "gestures on/off" (router `ui` intents). Gesture actions log one
  chat line ("Opened YouTube", "Closed Notepad"); nothing is spoken. The orb
  shows a small cyan dot while the camera is on.
- `setup.py hands` — downloads the model (~8 MB) to `models/`.
- `requirements-hands.txt` — `mediapipe` (brings OpenCV). Optional: voice-only
  installs stay as they were.

## Tiles

YouTube, Gmail, GitHub (sites); Chrome, Terminal, VS Code, Claude (apps);
adi, Desktop, Downloads (folders in Explorer). Fixed list in code; each
goes through the same `Workspace` method as the voice command.

## Errors

- Packages or model missing → the checkbox unticks and the chat says what
  to install / run.
- Camera missing or busy, or the camera thread fails → chat note, camera
  off; the rest of the assistant keeps running.
- An action fails (`ToolError`, Windows refuses focus) → chat note.

## Testing

- `tests/test_gestures.py` — synthetic hands: palm hold opens the menu,
  hover sectors, pinch picks, fist closes, swipe directions, two-finger
  swipe, thumbs-down hold/cancel, cooldown, jitter doesn't fire anything.
- `tests/test_windows.py` — fake `user32`: shell and own windows refused,
  Alt-Tab filtering, desktop keys are exactly Win+Ctrl+Arrow.
- `tests/test_tools.py` — Claude and Explorer argv via the fake launcher;
  Explorer refuses non-root names.
- `tests/test_router.py` — camera on/off phrases.
- Live camera test by the user.

## Safety notes

- Window control is new power, kept narrow: close (polite, app may refuse),
  switch window, switch desktop. No typing, no clicks, no mouse.
- The window title is shown in the overlay and chat only; it never reaches
  the LLM, the classifier, an action, or memory.
- Camera off by default; the Windows camera light and the orb dot show it's on.

## Changes after the phase review (2026-10-14)

The reviewer agent's findings and what changed:

1. Stop then quick start could leave the camera running unseen: each camera
   run now has its own token and engine; late callbacks from an old run are
   dropped.
2. Close could hit a different window than the one named: the window handle
   is captured when arming starts and `close_front(expected=...)` refuses if
   another window came to the front. A close with no named target does nothing.
3. Swipes fired on hand jumps and waves: losing the hand clears the track; a
   swipe needs ≥ 4 samples, no jump > 0.15 between frames, motion in one
   direction, and 0.28 frame widths (was 0.22).
4. Thumbs-down tuned on an impossible hand: MediaPipe's GestureRecognizer
   (trained on real hands) replaces the plain landmarker at the same cost;
   the fallback rule is rotation-free (thumb ≥ 0.5 hand sizes, within 35°
   of straight down, the lowest point).
5. A fist could read as a pick: pinch needs 3 frames, the other fingers
   clearly up (ratio 1.2), and the tile hovered for 0.25 s.
6. Thumbs-down could close the terminal running the assistant: our console
   and parent processes (up to, not including, Explorer) are protected from
   closing; switching to them is still allowed.
7. A refused `WM_CLOSE` (admin windows) now says so.
8. The package check no longer imports MediaPipe on the Tk thread, and
   voice "camera on" waits for the result before answering.

Also found while smoke-testing: quitting cleared the overlay after Tk was
destroyed, and the error skipped the rest of shutdown; fixed.

Second review round:

- A steady "hello" wave still switched windows about once a second: open-hand
  positions are now remembered through the cooldown, and a swipe is refused
  if the hand was moving the other way in the 0.4 s before it started. Cost:
  between two swipes, pause a moment after bringing the hand back.
- The overlay is built before the camera starts, so a Tk failure can't leave
  the camera running unseen.
- Thumbs down on a window that can't be closed (ours, the terminal we run in,
  the desktop) says "can't close this one" at once instead of counting down.
- A quick off/on waits up to 1 s for the old camera thread to let go.
- A pinch belongs to the tile it started on; sliding onto another tile while
  forming it cancels it until the fingers part.

## After the first live test (2026-10-14)

The first live try "barely" worked. A recording of hand points showed the
hand sitting at the bottom edge of the camera frame, and one held thumbs
down closing two windows. Changes, chosen with the user:

- Hand view: while the camera is on, a small click-through panel in the
  bottom-right corner draws the camera frame as a box, the hand's skeleton
  in it (points only) and the shape it reads; near an edge it says
  "raise your hand" / "move your hand to the middle".
- Thumbs down re-arms only after the thumb has been gone 0.6 s; one frame
  read as a fist mid-hold no longer counts as lowering it.
- The ring menu was too fiddly ("rotating that circle is really hard").
  It is now a 2x5 grid; the palm moves the highlight like a pointer, from
  the middle of the top row, one tile per 0.07 frame widths / 0.10 heights,
  with a quarter-tile margin against jitter. Open with a grab (fist, 3
  frames) or the barehands tap (quick pinch, let go within 0.6 s); any
  non-palm shape freezes the highlight. 👌 and "fist closes the menu" are
  gone; taking the hand away closes it.
- From barehands: tap (above) and flick, which our swipes already are.
  Clap, claw and two-hand scale move cards on its board and have no
  desktop equivalent; clap would also need two-hand tracking.
- New: one-finger (☝, MediaPipe "Pointing_Up") swipe switches tab in the
  front window with Ctrl+Tab / Ctrl+Shift+Tab, at the user's request; the
  keystroke rule in CLAUDE.md now lists it. Never sent to the desktop or
  taskbar. Swipes go by finger count: ✋ window, ☝ tab, ✌ desktop.

## Second live test: swipes (2026-10-14)

The grid worked ("works nicely now"); swipes never fired. A swipe recording
showed MediaPipe losing the hand for the fast part of nearly every stroke
(83 of 511 frames had a hand; left swipes were never seen on the left at
all). The room is dim: auto exposure runs a long shutter, and a short one
(1/64 s) gives a near-black picture even at the camera's maximum gain, so
the camera can't freeze the motion. Changes, chosen with the user
("keep swipes, fix for blur"; "tab dial"):

- Gap swipe: the hand lost at most 0.8 s and seen again >= 0.25 frame
  widths across (from where its visible motion began), same shape, size
  within 2x: a swipe. Replaying the recording: 2 right swipes, 0 wrong
  (1.1 s gaps picked up a swing back as a left swipe).
- Swing-back guard: a swipe opposite to the last one within 1.5 s is the
  hand returning and is ignored.
- Menu hold 0.6 -> 1.0 s: the hand rested ~0.6 s before swiping and the
  menu opened instead.
- MediaPipe thresholds lowered (detection 0.5, presence and tracking 0.4)
  to hold on to a blurred hand.
- Tabs: the one-finger swipe is replaced by a tab dial: hold one finger
  still 0.5 s, then each 0.07 frame widths of travel right / left sends one
  Ctrl+Tab / Ctrl+Shift+Tab (at most one per frame, quarter-step margin),
  slow enough for the camera to follow. A fist, the hand gone 0.6 s or 6 s
  idle ends it.
- Advice for the user: more light on the hand, and end a swipe with the
  hand held in view.

## Third live test: dials replace swipes (2026-10-14)

With more light the hand stayed in sight (419 of 459 frames), but no swipe
fired: the menu opened when the palm rested before the first swipe and then
took every swipe as pointer motion. The swipes themselves were slower than
the threshold (0.35 frame widths in 0.7-2 s). Replays with slower thresholds
fired on every swing back too. The user swipes, swings straight back,
rests, swipes again; the pause at the far end of a swing back (0.15-0.45 s)
overlapped the rest before a real swipe (0.4-0.55 s), so a pause rule gave
1 wrong switch and 2 misses out of 6, and one wrong switch flipped the
reading of the swipes after it. The tab dial in the same recording made 13
steps, none wrong. Chosen with the user ("Dial"):

- Window and desktop dials, like the tab dial: hold ✋ (windows) or ✌
  (desktops) still 0.5 s, then each 0.07 frame widths right / left is one
  step forward / back. Swipes and their blur and swing-back rules are gone.
- Menu: 👍 held 0.6 s (MediaPipe "Thumb_Up", or the thumb long, pointing up
  and the highest point), then the open palm moves the highlight. The
  highlight anchors once the palm has been open 3 frames, and grab / tap
  wait for it, so the thumbs up lowering through a fist picks nothing. No
  palm within 3 s of the thumb coming down closes the menu (live: the thumb
  stayed up 2.6 s after the menu opened, and the first version closed it). A thumbs up with the palm centre below
  0.88 doesn't count: the first recording read a hand cut off by the bottom
  edge as a thumbs up for over a second.
