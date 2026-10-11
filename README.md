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
- 🖥️ A window with an animated circuit-board face (idle, listening, thinking,
  speaking), a chat log, and a box to type in
- ✋ Hold **Right Ctrl** (from any window), Space, or the floating orb to talk:
  the mic is only used while you hold it, so it never answers background talk
  or its own voice. Hands-free mode is one tick (or "go hands free") away.
  Stop (or Esc) cuts a reply short; "stop" quits.
- 🔵 A small orb stays in the top-right corner of the screen, so you can see
  it's listening or thinking even with YouTube in front. Click it to bring the
  window back.
- 🗂️ Read-only desktop help: open Chrome (in the profile you name), the
  terminal, VS Code, Gmail, YouTube or GitHub; search YouTube, GitHub or
  Google; **play YouTube videos** ("play lofi on YouTube", "play the second
  one"); move between folders in `C:\adi`, Desktop and Downloads; list them;
  explain a folder or file. It never writes, deletes or runs anything.
- 📬 Optional read-only **Gmail** and **GitHub**: "how many emails did I get
  today?", "find the email from Amazon", "any open issues on voice AI?". See
  [SETUP_ACCOUNTS.md](SETUP_ACCOUNTS.md).
- 🧠 Memory: "remember that my exam is on Friday". Its only write, to its own
  `data/memory.json`.

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

```powershell
python -m assistant.main              # the window (default)
python -m assistant.main --text       # type in the terminal instead
python -m assistant.main --terminal   # old always-listening loop, no window
```

In the window:

- **Hold Right Ctrl** from any window (or Space in the window, or the orb, or
  the 🎤 button), talk, let go. Space types normally once the chat box has
  text. Ctrl shortcuts like Ctrl+C never start a recording.
- **F11** full screen. "Change your face" switches between the circuit board
  and the orb. "Hide" leaves just the corner orb.
- "What can you do?" lists everything.
- **Type** in the box and press Enter.
- **Esc** or **■ Stop** cuts the current reply short.
- Say or type **"stop"** (or "goodbye") to quit, or close the window.
- **Hands-free**: tick it to talk without holding a key. Use headphones: on
  open speakers it may still pick up other voices in the room.

Things to try:

- "Open Chrome." / "Open Chrome as *profile name*." / "Open Gmail." / "Open a terminal here."
- "Search YouTube for lofi music." / "Look up voice assistants on GitHub." /
  "Google pasta recipes."
- "Play Coldplay on YouTube." then "Play the next one." / "Search YouTube for
  anime", "What are the videos?", "Play the second one."
- "What's in the downloads folder?" / "What time is it in Tokyo?" / "What's
  the weather in Paris?" (opens Google's answer) / "Remember that my exam is
  on Friday." / "Start over."
- "Go to the voice input folder." / "Go up." / "Where am I?"
- "What's in this folder?" / "Explain this project."
- "What does main.py do?" — then "What else does it do?"
- "cd voice" — if two folders match, it asks which; answer "the first one".

## Tests

```powershell
python -m unittest tests.test_tools tests.test_router tests.test_classifier tests.test_conversation tests.test_audio tests.test_ui tests.test_youtube tests.test_accounts
python -m tests.eval_classifier   # live check of the LLM classifier (needs Ollama)
```

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

Built in 15 phases: echo bot → brain → streaming → fallback → polish →
desktop tools → fast path → LLM fallback → conversation polish → live-test
fixes → window → Chrome profiles and search → second test's fixes and
YouTube playing → corner orb, Right Ctrl and voice commands → memory, Gmail
and GitHub.
Each phase is reviewed before the next begins.
