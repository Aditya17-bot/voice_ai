# Architecture

Offline, real-time, audio-in / audio-out conversational assistant.

## Pipeline

```
┌─────┐   ┌─────┐   ┌──────────────┐   ┌───────────┐   ┌───────┐   ┌─────────┐
│ Mic ├──►│ VAD ├──►│ STT          ├──►│ LLM       ├──►│ TTS   ├──►│ Speaker │
│     │   │     │   │ faster-      │   │ Ollama    │   │ Piper │   │         │
│     │   │silero│  │ whisper      │   │ 3B, local │   │       │   │         │
└─────┘   └─────┘   └──────────────┘   └───────────┘   └───────┘   └─────────┘
  sounddevice         speech→text        streaming        text→wav    sounddevice
```

One Python process. Each stage runs on its own thread; stages communicate
through `queue.Queue`. This keeps the design simple (no asyncio) while letting
audio capture, LLM generation, and playback overlap.

## Stages

### 1. Mic capture (`audio_io.py`)
- `sounddevice.InputStream`, 16 kHz mono int16, 30 ms frames.
- Frames pushed onto the VAD queue.

### 2. VAD (`vad.py`)
- Silero VAD (`pysilero-vad`, ONNX runtime — no torch dependency) scores each
  512-sample chunk for speech probability.
- State machine: IDLE → SPEAKING (speech detected) → END (≥ ~500 ms trailing
  silence). On END, the buffered utterance is sent to STT.

### 3. STT (`stt.py`)
- `faster-whisper`, `base` model, `int8` compute on CPU.
- Transcribes the complete utterance in one shot (~300 ms for a short query).

### 4. LLM (`brain.py`)
- POST to Ollama `/api/chat` with `stream: true`.
- Keeps a rolling conversation history (last ~10 turns).
- Tokens are accumulated and split at sentence boundaries (`.`, `?`, `!`).
  Each complete sentence is pushed to the TTS queue immediately.

### 5. TTS (`tts.py`)
- Piper synthesizes each sentence to 22 kHz PCM.
- Audio chunks are queued to a dedicated playback thread.

### 6. Fallback engine (`fallback.py`)
- Watchdog timer starts when STT text is sent to the LLM.
- If no first sentence within **1.2 s** → play a filler clip
  ("Hmm, let me think about that…"). Fillers are pre-generated with Piper and
  stored in `fillers/`; rotation avoids immediate repeats.
- If Ollama is unreachable / errors → canned conversational responses
  ("I'm having trouble gathering my thoughts — could you say that again?"),
  never a raw error.

## Latency budget (target ≤ 2 s to first spoken word)

Measured on the dev machine (RTX 3050 Laptop 4 GB, llama3.2:3b, `num_ctx` 2048):

| Stage | Budget | Measured |
|---|---|---|
| VAD end-of-speech detection | ~600 ms (trailing-silence window) | 600 ms by design |
| STT (whisper base, int8) | ~300 ms | ~600 ms |
| LLM first sentence (3B, streamed) | ~600 ms | ~570–1160 ms |
| TTS first sentence (Piper) | ~200 ms | ~100–540 ms |
| LLM + TTS to first audio | — | **mean ~1.34 s** |

Desktop actions, measured from the end of the transcript (add VAD + STT):

| Path | First word |
|---|---|
| Regex fast path (open, go to, list) | ~0.1–0.3 s |
| Classifier picks an action | ~0.6–0.7 s |
| Explain a file (fast path / via classifier) | ~1.3 s / ~2.1 s |
| Classifier says "just talk", then a reply | ~1.6 s |
| "Want that one?" offer / "Yes" then explanation | ~0.2 s / ~1.8 s |
| Follow-up question about the last file | ~1.5 s |

Explaining a file through the classifier takes two LLM calls and runs a
little over budget; the filler covers it. The filler covers any case where
the LLM stage blows its budget.

Hard-won latency notes:
- **Use `127.0.0.1`, never `localhost`**, for the Ollama URL — on Windows the
  IPv6-first resolution of `localhost` adds ~2 s to every request.
- **`num_ctx: 2048`** shrinks the KV cache so llama3.2:3b runs 100 % on a
  4 GB GPU; at the default 4096 it spills 20 % to CPU and slows ~3x.
- 1b vs 3b barely matters once fully on GPU (1.21 s vs 1.34 s mean first
  audio) — keep 3b for answer quality.

### 7. Interrupting a reply
- **In the window (default):** push-to-talk. Holding Space or the mic button
  stops the reply and records; Esc or ■ Stop just stops. The mic stream stays
  open, but audio is discarded unless the key is held (or hands-free is on and
  the assistant is quiet), so the assistant never transcribes its own voice
  or background talk.
- **`--terminal --barge-in` (`vad.py:monitor_barge_in`):** the mic is watched
  while the reply plays, at a stricter threshold (≥ 0.75 sustained 320 ms).
  Only usable with headphones: in the first live test on laptop speakers, 13
  of the replies were cut off by the bot's own voice. Off by default.
- `Assistant.interrupt()` empties the player, cancels fillers, and makes the
  turn drop the rest of the streamed reply.

### Playback (`audio_io.Player`)
One long-lived `sd.OutputStream` with a callback that pulls from a queue of
clips. Sentences play back to back with no gap, `stop()` only empties the
queue under a lock, and "done" is reported once the output latency has
passed, so the mic drain that follows also drops the echo of the last word.
The earlier player called `sd.play`/`sd.wait` per sentence and `sd.stop()`
from another thread; that cross-thread use of sounddevice's global stream is
the likely cause of a segfault (exit 139) in the first live test.
`faulthandler` is on, so any further native crash prints a stack.

### The window (`ui.py`, `face.py`)
Plain tkinter, no extra dependency. Tk's main loop owns the widgets; a worker
thread runs one request at a time (transcribe, then `Assistant.respond`) and
a mic thread handles push-to-talk and hands-free listening. Both only talk
to the window through a queue that the main loop polls every 30 ms. The face
is a procedural circuit board whose pulses change with the state (idle,
listening, thinking, speaking) and with the loudness of the voice. The idea
of a face that idles, listens, thinks and speaks, and of holding a key to
talk, comes from jaredrhod's fullstack-agent; the code is our own.

### 8. Desktop actions — read-only (`tools.py`, `router.py`, `classifier.py`)

The assistant can open a few apps and sites and browse, list, and explain
folders and files. It can **never write, delete, move, or run anything**.

**What it can touch**
- Folders: `C:\adi`, Desktop (OneDrive Desktop if present), Downloads.
  Every path is resolved and must stay inside one of them; links leading
  outside are hidden. Dot-folders like `.git` and clutter like
  `node_modules` can't be entered or read. Secrets (`.env`, `.ssh`, keys, `*credential*`, …) are
  invisible, and private-key text is refused even under an innocent name.
- Apps: Chrome, Windows Terminal (or PowerShell), VS Code, launched by
  absolute path with an argument list, never through a shell.
- Sites: Gmail, YouTube, GitHub. There is no "open any URL", so nothing can
  be sent anywhere.
- Searches: YouTube, GitHub and Google results pages. The query is the
  user's own words, taken by the regex router and URL-encoded, capped at 120
  characters. The LLM classifier has no search option, and file text never
  reaches a query.
- Chrome opens in a fixed profile (`Default`) so the profile picker never
  shows; "open Chrome as <name>" picks another by the display name in
  Chrome's `Local State` (read-only, names only).

**How an utterance is handled**

```
utterance
  ├─ short answer to "did you mean X or Y?" / "want that one?" ──► open it
  │    ("yes", "the first one", "voice input"; anything else is a new request)
  ├─ follow-up on the file just explained ──► grounded reply
  ├─ regex fast path (router.py) ─────► act, ~0.1–0.3 s
  │    explicit ("cd tests", "read main.py"): fuzzy names, say if missing
  │    loose ("go to X", "explain X"): exact names only, else fall through
  ├─ sounds like a desktop request? ──► LLM classifier (classifier.py)
  │    one constrained-JSON call, ~0.27 s: pick one option from a menu
  │    ("open gmail", "go to tests", "read main.py", "none"), then check the
  │    choice is backed by the user's words; otherwise just talk
  └─ conversation ────────────────────► streamed LLM reply
```

Reading a file or folder doesn't speak its text: the text goes to the LLM
between `<<<DATA>>>` markers, marked as data rather than instructions, and
the LLM explains it in a few sentences. A follow-up anchored to that file
("what else does it do?", "tell me more about the code") in the next two
turns reuses the text; small talk with "it" ("what time is it?") doesn't.
Any other action (moving, listing, opening an app) or an error ends the
thread.

If a file isn't in the current folder but exactly one subfolder has it,
the assistant offers it ("main.py isn't here, but there's one in
assistant. Want that one?"). The search compares names only, skips hidden
and private entries, and gives up after 0.15 s. A loosely phrased question
gets the offer only if the name has an extension, so "what does docker
compose do?" stays a general question.

**Why the LLM classifies instead of calling tools.** With native tool
calling, llama3.2:3b called a tool on most chit-chat that passed the gate
("go to sleep" → open Desktop). The classifier's output is limited by
Ollama to one menu entry, so it can't invent a file name or a URL, and
nothing it writes is spoken. It sees only the utterance and the names here,
with no history and no file text, so a file can't steer an action. A choice
is still dropped unless the user named the target in whole words, used an
opening verb for apps and sites, and didn't say "don't" or "later".
qwen3:4b was tried and rejected: the local build ignores `think: false` and
spends 6–49 s reasoning.

Measured with `python -m tests.eval_classifier` (llama3.2:3b): 27/29 on the
tuning set, 18/20 on a held-out set, 0 wrong actions on either.

## Phases (all complete)

1. **Echo bot** — mic → VAD → STT → TTS → speaker. Proves the audio loop.
2. **Brain** — insert Ollama; blocking (full reply) first.
3. **Streaming** — token streaming + sentence-level TTS pipelining.
4. **Fallback** — filler watchdog, canned responses, no generic errors.
5. **Polish** — barge-in (interrupt TTS when user speaks), history trimming,
   model-size tuning.
6. **Desktop tools** — read-only sandboxed folder browsing, file reading,
   allowlisted apps and sites; `--text` mode for typed testing.
7. **Fast path** — regex router for common commands; grounded explanations
   of files and folders.
8. **LLM fallback** — constrained-JSON classifier for requests the router
   misses, with word-backing checks.
9. **Conversation polish** — answering "did you mean", follow-up questions
   about the last file, names spoken naturally ("main dot py").
10. **Live-test fixes** — persistent output stream (gapless, no cross-thread
    sounddevice calls), barge-in off by default, the model no longer claims
    to have looked at files, "check X", spoken "stop" quits.
11. **Window** — tkinter face, chat log, typing, push-to-talk, hands-free,
    Stop.
12. **Chrome profiles and search** — open Chrome as a named profile; search
    YouTube, GitHub, Google with the user's own words.
13. **Second live test's fixes, YouTube playing** — time and date, "what can
    you do", "what's in the downloads folder", "the folder named X", video
    requests search YouTube, push-to-talk keeps 0.3 s before the press,
    Whisper `base.en`, and "play X on YouTube" / "play the second one".
14. **Corner orb, Right Ctrl, voice console** — an always-on-top orb shows
    the state over other apps; hold Right Ctrl from any window to talk; "go
    hands free", "full screen", "change your face", "start over"; a second
    face (radial orb).
15. **Memory, Gmail, GitHub** — "remember that…" (the one write, to
    `data/memory.json`); read-only Gmail and GitHub with server-enforced
    read-only logins.
16. **Hand gestures** — webcam hand control: a thumbs-up menu of apps,
    sites and folders, dials (hold a shape, then move) to switch windows,
    desktops and tabs, thumbs down to close the front window. The Claude desktop app
    joins the app list. Design: `docs/superpowers/specs/2026-10-14-hand-gestures-design.md`.

### YouTube playing (`youtube.py`)
The assistant can't click inside Chrome, so it reads YouTube's results page
for the user's words (with `requests`, as a browser would), takes the video
ids from the page's `ytInitialData`, and opens
`https://www.youtube.com/watch?v=<id>`. Only youtube.com is fetched or
opened, only ids matching `[A-Za-z0-9_-]{11}` are used, and titles are only
spoken and shown. A search fetches its results in the background, so "play
the second one" is instant; a direct "play X" takes 1–4 s and says "Looking
for X" first. If YouTube's page layout changes, only the top id is trusted.
The YouTube Data API would be the official route but needs an API key.

### Accounts (`gmail.py`, `github.py`, `credentials.py`, `setup.py`)
Gmail uses the `gmail.readonly` OAuth scope and GitHub a fine-grained
read-only token, so the servers refuse any change. Setup refuses anything
broader. Logins are DPAPI-encrypted in `%APPDATA%\voice_assistant`. Only GET
requests are sent (plus Google's token refresh). Counts and lists (sender,
subject, when) are spoken directly; an email body or a README goes to the LLM
as delimited data in a tool-less turn, like a file, and follow-ups work the
same way. Email and repo text never reaches the classifier or any action.

### Time, weather (`clock.py`)
Local time and date, and "what time is it in Tokyo", are worked out from the
`tzdata` time-zone database (the 3B model got time-zone sums wrong). Weather
can't be known offline, so "what's the weather in Paris" opens Google's
answer for those words.

### Hand gestures (`hands.py`, `gestures.py`, `handmenu.py`, `windows.py`)
Off until "✋ Hands" is ticked or you say "camera on". A camera thread grabs
mirrored frames (15 fps, 5 fps after 3 s with no hand) and MediaPipe's
gesture recognizer turns each into 21 points plus a label (Open_Palm,
Victory, Thumb_Down, …) from a model trained on real hands; frames are never
saved or sent. `gestures.py` is a pure state machine from points and labels
to events (point rules are the fallback when the label has no opinion), so
it is unit-tested without a camera:

| Gesture | Does |
|---|---|
| 👍 thumbs up held 0.6 s | hand menu: a 2×5 grid; then the open palm moves the highlight like a pointer; ✊ grab or a quick pinch (tap) opens it; hand away closes |
| ✋ open hand held still 0.5 s, then moved | window dial: every 0.07 frame widths right / left is the next / previous window (like Alt+Tab) |
| ✌ two fingers held still 0.5 s, then moved | desktop dial: right / left virtual desktop |
| ☝ one finger held still 0.5 s, then moved | tab dial: next / previous tab in the front window |
| 👎 thumbs down held 1 s | close the front window (a red ring fills; lower the hand to cancel) |

A dial is a scroll wheel: steps follow where the hand is from where the
dial opened, at most one per frame, only once the hand is a quarter step
past the line (no jitter steps), so moving back steps back. Any other shape
holds it; ✊, hand away 0.6 s or 6 s without a step ends it. Every action
and every dial's end is followed by 1 s of cooldown. In the menu the
highlight starts on the middle of the top row once the palm has been open
3 frames (the thumbs up lowering through a fist is not a grab); the palm
moves it one tile per 0.07 frame widths (0.10 heights), and only once it is
a quarter tile past the border, so jitter never flickers. No open palm
within 3 s of the thumb coming down closes the menu. Any shape but an open palm freezes the
highlight, so closing the hand doesn't move it. A grab is a fist for 3
frames; a tap (from barehands) is a pinch of 2+ frames let go within 0.6 s;
either opens the tile only if the highlight had stayed on it 0.25 s when the
hand began to close. The first
version (a ring of tiles picked by the palm's angle, 👌 to open) was too
fiddly in the live test. Menu tiles go through the same `Workspace` methods as
voice: YouTube, Gmail, GitHub, Chrome, Terminal, VS Code, Claude, and the
allowed root folders, which open in File Explorer (only those roots).

Window control (`windows.py`, ctypes) is kept narrow: close posts
`WM_CLOSE`, the same as the window's X, so the app still asks to save. It
closes only the window the overlay named when the thumb went down (if
another came to the front, nothing closes), and refuses the desktop, the
taskbar, our own windows and the terminal or editor we were started from. Switching brings an
Alt-Tab window (this desktop only) to the front; "next" also sends the old
one to the back, so repeated steps cycle, and a step back undoes one. The only keystrokes ever sent
are Win+Ctrl+Left/Right and Ctrl+Tab / Ctrl+Shift+Tab (never to the
desktop or taskbar), with the left Ctrl so the talk key never sees them.
Window titles appear in the overlay and the chat only.

The overlay is a click-through, never-focused, see-through window in the
middle of the screen; it stays shown and simply draws nothing when idle,
because showing a window takes the focus. Optional install:
`pip install -r requirements-hands.txt` and `python -m assistant.setup hands`
(~8 MB model). Measured: 16–23 ms per frame on the CPU; the camera takes
~1.6 s to open. `python -m assistant.hands [seconds]` is a dry run for
tuning the thresholds in `gestures.py`: it prints the label, hand shape,
thumb–index gap and every gesture event, and acts on nothing. Desktop
switching was checked live (temporary second desktop, both directions).

First live try (2026-10-14) barely worked. A recording of the hand points
showed why: the hand sat at the bottom edge of the camera frame (palm
centre y ≈ 0.95), so MediaPipe saw part of it and kept losing it, and the
user had no way to know. Since then, while the camera is on, a small hand
view in the bottom-right corner draws the frame as a box, the hand's
skeleton in it (points only, never the picture) and the shape it reads,
or "raise your hand ↑" / "move your hand to the middle" near the edges.
The same recording showed a held thumbs down closing two windows: a frame
whose label dipped under the score cut read as a fist and counted as
"thumb lowered". The thumb must now be gone 0.6 s (`CLOSE_REARM_S`).

Windows and desktops were swipes until the second and third live tests. In
a dim room the camera's long exposure blurred a fast hand out of sight for
the whole stroke (allowed for by counting a hand that vanished and came back
across the frame). In good light the hand stayed in sight, but recordings
showed swipe, swing straight back, rest, swipe again, with the pause at the
far end of a swing back (0.15–0.45 s) no shorter than the rest before a
real swipe (0.4–0.55 s): every rule tried either switched windows on the
swing back or missed real swipes, and one wrong switch made the next real
swipe look like a swing back. The same recording ran the tab dial 13 steps
with none wrong, so windows and desktops became dials too. The menu moved
from a still palm to a thumbs up, because a palm resting between swipes
opened it; a thumbs up read on a hand cut off by the bottom of the frame
(palm centre below 0.88) doesn't count.

### Right Ctrl talk key (`hotkey.py`)
No keyboard hook: a thread polls `GetAsyncKeyState` for Right Ctrl every
20 ms and starts recording after 200 ms held. While it's held, any other key
or mouse button marks it as a shortcut (Ctrl+C, Ctrl+scroll), and the
recording is dropped. Nothing is recorded or logged about other keys.

Tried and dropped in phase 10: Whisper `hotwords` for app names. It fixed
"search your tube", but silence then took ~23 s to transcribe and loud noise
came out as "www.cf.com". Push-to-talk solved more of the mishearing.

## Non-goals

- No web server, no cloud AI: speech, the model and the voice stay local.
  The only network use is what the user asks for (opening sites, YouTube
  results, their own Gmail and GitHub).
- No wake word (push-to-talk is enough).
- No typing into other apps: keystrokes could land in a terminal and run.
  (The fixed exceptions: the desktop switch and Ctrl+Tab / Ctrl+Shift+Tab.
  Neither contains Enter or text, so in a terminal nothing can run.)
- No mouse control by hand: a wrong click is worse than a miss.
