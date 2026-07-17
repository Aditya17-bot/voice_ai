# Offline Voice Assistant

Real-time audio-in / audio-out conversational AI. Fully offline: speech
recognition, language model, and speech synthesis all run on your machine.

## Features

- 🎤 Voice input with automatic end-of-speech detection (Silero VAD)
- 🧠 Local LLM via Ollama (llama3.2 3B)
- 🔊 Natural voice output (Piper TTS)
- ⚡ Streams the answer sentence-by-sentence — first words in ~1.5 s
- 🛟 Graceful fallback: filler phrases while thinking, canned conversation if
  the model stalls — never a generic error message
- ✋ Barge-in: speak over the bot and it stops and listens (best with
  headphones — open speakers can leak the bot's voice into the mic)

## Requirements

- Windows 10/11 (works on Linux/macOS too), Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~4 GB free RAM for the 3B model (CPU is fine; GPU is faster)

## Setup

```powershell
# 1. Python deps
pip install -r requirements.txt

# 2. LLM (needs Ollama running)
ollama pull llama3.2:3b

# 3. TTS voice (Piper) — downloaded automatically on first run,
#    or manually into models/piper/

# 4. Run
python -m assistant.main
```

## Usage

Speak when you see `🎤 Listening…`. Pause, and the assistant answers out loud.
Press `Ctrl+C` to exit.

## How it works

```
Mic → VAD → STT (faster-whisper) → LLM (Ollama) → TTS (Piper) → Speaker
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for design details and the latency
budget.

## AI use disclosure

AI (Anthropic's Claude, via Claude Code) was used in writing the code in this
project: the pipeline modules under `assistant/`, the tests, and the project
documentation (README, ARCHITECTURE) were written with AI assistance under
human direction, including debugging and latency optimization. Design
decisions, phase planning, and final verification (live microphone testing)
were done by the author. See `AI_USE.md` for details.

## Project status

Built in 5 phases: echo bot → brain → streaming → fallback → polish.
Each phase is reviewed before the next begins.
