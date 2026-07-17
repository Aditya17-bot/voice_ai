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

The filler covers the case where the LLM stage blows its budget.

Hard-won latency notes:
- **Use `127.0.0.1`, never `localhost`**, for the Ollama URL — on Windows the
  IPv6-first resolution of `localhost` adds ~2 s to every request.
- **`num_ctx: 2048`** shrinks the KV cache so llama3.2:3b runs 100 % on a
  4 GB GPU; at the default 4096 it spills 20 % to CPU and slows ~3x.
- 1b vs 3b barely matters once fully on GPU (1.21 s vs 1.34 s mean first
  audio) — keep 3b for answer quality.

### 7. Barge-in (`vad.py:monitor_barge_in`)
- While the reply plays, the mic keeps being monitored.
- Stricter than normal listening (probability ≥ 0.75 sustained 320 ms) because
  the bot's own voice leaks from speakers into the mic.
- On barge-in: `Player.stop()` flushes playback, and the captured speech
  chunks seed the next `listen()` so the user's first words aren't lost.
- Reliable with headphones; on open laptop speakers the bot can occasionally
  trigger on itself (no echo cancellation — accepted simplicity tradeoff).

## Phases (all complete)

1. **Echo bot** — mic → VAD → STT → TTS → speaker. Proves the audio loop.
2. **Brain** — insert Ollama; blocking (full reply) first.
3. **Streaming** — token streaming + sentence-level TTS pipelining.
4. **Fallback** — filler watchdog, canned responses, no generic errors.
5. **Polish** — barge-in (interrupt TTS when user speaks), history trimming,
   model-size tuning.

## Non-goals

- No GUI, no web server, no cloud calls.
- No wake word (push-to-talk / always-listening loop is enough).
