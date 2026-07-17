# AI Use Disclosure

This project was built with the assistance of AI, used transparently as follows.

## Where AI was used

- **Writing code** — the primary use. All Python modules in `assistant/`
  (audio capture, voice activity detection, speech-to-text, LLM integration,
  text-to-speech, fallback engine, barge-in).

## Where AI was not used

- Project requirements and acceptance criteria.
- Final live microphone testing and acceptance of the working system.
- The decision of which architecture trade-offs to accept (offline-first,
  no echo cancellation, model choice within the latency budget).

## Runtime AI components (offline, local)

The assistant itself runs three local AI models at runtime, all offline:
Silero VAD (speech detection), faster-whisper `base` (speech-to-text),
llama3.2:3b via Ollama (language model), and Piper (speech synthesis).
No cloud AI services are called at runtime.
