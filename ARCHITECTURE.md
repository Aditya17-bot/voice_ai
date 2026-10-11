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

### 7. Barge-in (`vad.py:monitor_barge_in`)
- While the reply plays, the mic keeps being monitored.
- Stricter than normal listening (probability ≥ 0.75 sustained 320 ms) because
  the bot's own voice leaks from speakers into the mic.
- On barge-in: `Player.stop()` flushes playback, and the captured speech
  chunks seed the next `listen()` so the user's first words aren't lost.
- Reliable with headphones; on open laptop speakers the bot can occasionally
  trigger on itself (no echo cancellation — accepted simplicity tradeoff).

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

## Non-goals

- No GUI, no web server, no cloud calls.
- No wake word (push-to-talk / always-listening loop is enough).
