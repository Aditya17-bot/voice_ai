# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Offline real-time voice assistant: Mic → VAD → STT → LLM → TTS → Speaker.
Target: end-to-end response under 2 seconds, with a filler/fallback flow when the LLM is slow or unavailable.

## Stack (all offline, all local)

- **Audio I/O**: `sounddevice` (16 kHz mono mic capture, playback)
- **VAD**: `pysilero-vad` (Silero model on onnxruntime, no torch) — detects end of user speech
- **STT**: `faster-whisper` (base/small model, CPU int8)
- **LLM**: Ollama at `localhost:11434`, model `llama3.2:3b` (fallback `qwen2.5:1.5b` on weak CPU)
- **TTS**: Piper (local neural TTS, per-sentence synthesis)

## Architecture rules

- Single Python process, thread-per-stage, queues between stages. No async frameworks.
- Stream LLM tokens; split on sentence boundaries; synthesize and play sentence-by-sentence. Never wait for the full LLM reply before speaking.
- Fallback: if first sentence is not ready within ~1.2 s, play a rotating pre-recorded filler clip. Never play the same filler twice in a row. If the LLM is unreachable, use canned conversational responses — never a generic error message.
- Keep it simple: no web server, no GUI, terminal app only.

## Layout

- `assistant/` — package source (one module per pipeline stage)
- `assistant/main.py` — entry point, wires stages together
- `models/` — downloaded model files (gitignored)
- `fillers/` — pre-generated filler WAV clips
- `ARCHITECTURE.md` — design details and latency budget

## Phases

Work proceeds in 5 phases (see ARCHITECTURE.md). After each phase is complete, a reviewer agent criticizes the phase and recommends alternatives before moving on.

## Conventions

- Python 3.10+, type hints on public functions.
- Run: `python -m assistant.main`
- No heavyweight dependencies beyond the stack listed above.
