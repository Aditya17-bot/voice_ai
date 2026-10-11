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
- Keep it simple: no web server, no GUI, terminal app only.
- Desktop actions are READ-ONLY: no writing, deleting, moving, or running commands; no arbitrary URLs. Paths stay inside the allowed roots. Apps and sites are fixed allowlists. A wrong action is worse than a miss.
- File and folder text goes to the LLM only as delimited data in a tool-less turn; the LLM that chooses actions (the classifier) never sees file text or history.

## Layout

- `assistant/` — package source (one module per pipeline stage)
- `assistant/main.py` — entry point, wires stages together; `Assistant` decides how each utterance is handled
- `assistant/tools.py` — read-only sandbox: folders, files, app/site launching
- `assistant/router.py` — regex fast path for spoken commands, follow-up helpers
- `assistant/classifier.py` — constrained-JSON LLM fallback for commands the router misses
- `tests/` — stdlib `unittest` suites; `eval_classifier.py` is a live eval needing Ollama
- `models/` — downloaded model files (gitignored)
- `fillers/` — pre-generated filler WAV clips
- `ARCHITECTURE.md` — design details and latency budget

## Phases

Work proceeds in phases (9 so far, see ARCHITECTURE.md). After each phase is complete, a reviewer agent criticizes the phase and recommends alternatives before moving on.

## Conventions

- Python 3.10+, type hints on public functions.
- Run: `python -m assistant.main` (`--text` to type instead of talk)
- Test: `python -m unittest tests.test_tools tests.test_router tests.test_classifier tests.test_conversation`
- No heavyweight dependencies beyond the stack listed above.
