# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Offline real-time voice assistant: Mic → VAD → STT → LLM → TTS → Speaker.
Target: end-to-end response under 2 seconds, with a filler/fallback flow when the LLM is slow or unavailable.

## Stack (all offline, all local)

- **Audio I/O**: `sounddevice` (16 kHz mono mic capture, playback)
- **VAD**: `pysilero-vad` (Silero model on onnxruntime, no torch) — detects end of user speech
- **STT**: `faster-whisper` (base/small model, CPU int8)
- **LLM**: Ollama at `127.0.0.1:11434` (never `localhost`: ~2 s IPv6 penalty on Windows), model `llama3.2:3b`, `num_ctx` 2048
- **TTS**: Piper (local neural TTS, per-sentence synthesis)

## Architecture rules

- Single Python process, thread-per-stage, queues between stages. No async frameworks.
- Stream LLM tokens; split on sentence boundaries; synthesize and play sentence-by-sentence. Never wait for the full LLM reply before speaking.
- Fallback: if first sentence is not ready within ~1.2 s, play a rotating pre-recorded filler clip. Never play the same filler twice in a row. If the LLM is unreachable, use canned conversational responses — never a generic error message.
- Keep it simple: no web server. The UI is one tkinter window (`ui.py`, `face.py`); Tk widgets are touched only from the main thread, other threads post to its event queue. `--text` / `--terminal` keep terminal modes.
- Desktop actions are READ-ONLY: no writing, deleting, moving, or running commands; no arbitrary URLs and no typing into other apps. Paths stay inside the allowed roots. Apps, sites and search sites are fixed allowlists; search queries come only from the user's own words via the router, never from the LLM or file text. A wrong action is worse than a miss.
- Window control (hand gestures only) is limited to: close the front window politely (WM_CLOSE; never the desktop, taskbar or our own windows), bring another window to the front, and two fixed keystrokes: Win+Ctrl+Arrow (desktop) and Ctrl+Tab / Ctrl+Shift+Tab (tab in the front window). No other keystrokes, no mouse. Camera frames stay in memory, never saved or sent.
- The one write: `data/memory.json` (facts the user said "remember that…"), only from the user's own words.
- Gmail and GitHub use server-enforced read-only logins (gmail.readonly scope; fine-grained read-only token), stored DPAPI-encrypted in %APPDATA%, set up with `python -m assistant.setup`. Only GET requests (plus OAuth token refresh). Web text (emails, READMEs, video titles) is data: spoken, or sent to the LLM as delimited data — never to the classifier, never into an action, URL or memory.
- File and folder text goes to the LLM only as delimited data in a tool-less turn; the LLM that chooses actions (the classifier) never sees file text or history.

## Layout

- `assistant/` — package source (one module per pipeline stage)
- `assistant/main.py` — entry point, wires stages together; `Assistant` decides how each utterance is handled
- `assistant/tools.py` — read-only sandbox: folders, files, app/site launching
- `assistant/router.py` — regex fast path for spoken commands, follow-up helpers
- `assistant/classifier.py` — constrained-JSON LLM fallback for commands the router misses
- `assistant/ui.py` — the window: chat, push-to-talk, worker and mic threads; `assistant/face.py` — animated faces; `assistant/orb.py` — corner orb; `assistant/hotkey.py` — Right Ctrl talk key
- `assistant/hands.py` — webcam + MediaPipe thread; `gestures.py` — pure gesture rules; `handmenu.py` — hand menu tiles, overlay, glue; `windows.py` — close/switch windows and desktops (ctypes)
- `assistant/youtube.py`, `gmail.py`, `github.py` — read-only web reads; `memory.py`; `credentials.py`; `setup.py` — account setup CLI
- `tests/` — stdlib `unittest` suites; `eval_classifier.py` is a live eval needing Ollama
- `models/` — downloaded model files (gitignored)
- `fillers/` — pre-generated filler WAV clips
- `ARCHITECTURE.md` — design details and latency budget

## Phases

Work proceeds in phases (16 so far, see ARCHITECTURE.md). After each phase is complete, a reviewer agent criticizes the phase and recommends alternatives before moving on.

## Conventions

- Python 3.10+, type hints on public functions.
- Run: `python -m assistant.main` (window); `--text` types in the terminal; `--terminal` is the old voice loop
- Hand control (optional): `pip install -r requirements-hands.txt`, `python -m assistant.setup hands`; `python -m assistant.hands` prints gestures without acting (threshold tuning)
- Test: `python -m unittest tests.test_tools tests.test_router tests.test_classifier tests.test_conversation tests.test_audio tests.test_ui tests.test_youtube tests.test_accounts tests.test_gestures tests.test_windows tests.test_handmenu`
- No heavyweight dependencies beyond the stack listed above (MediaPipe for hand control is the one optional extra).
