"""LLM stage: local Ollama chat with rolling history."""

import json
import re
from typing import Iterator

import requests

# 127.0.0.1, not localhost: on Windows, localhost tries IPv6 first and the
# fallback to IPv4 adds ~2 s to every request.
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3.2:3b"  # with num_ctx 2048 it runs 100% GPU on a 4 GB card; ~1.3 s to first audio
MAX_HISTORY_TURNS = 10  # user+assistant message pairs kept in context

# Small context keeps the whole model in VRAM on small GPUs and speeds up
# prompt evaluation. Spoken turns are short; 2048 tokens is plenty.
OLLAMA_OPTIONS = {"num_ctx": 2048}

# Sentence ends at . ! or ? followed by whitespace. Decimal points ("3.5")
# never match because they lack the trailing whitespace. Trailing text (a
# sentence still being generated) stays in the remainder until more tokens
# arrive.
_SENTENCE_END = re.compile(r"(.+?[.!?])(?:\s+|$)(?=\S|$)")
_ABBREVIATION = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|approx|e\.g|i\.e|U\.S|U\.K)\.$",
    re.IGNORECASE,
)


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off complete sentences; return (sentences, remainder)."""
    sentences: list[str] = []
    pending = ""
    pos = 0
    for match in _SENTENCE_END.finditer(buffer):
        # A match running to the buffer end may still be extended by the
        # next token — leave it in the remainder.
        if match.end() >= len(buffer):
            break
        candidate = (pending + match.group(1)).strip()
        pos = match.end()
        if _ABBREVIATION.search(candidate):
            # "Mr." etc. is not a sentence end — glue it to the next chunk.
            pending += buffer[match.start() : match.end()]
            continue
        pending = ""
        if candidate:
            sentences.append(candidate)
    return sentences, pending + buffer[pos:]


SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies are spoken aloud, so: "
    "answer in 1-3 short conversational sentences, no markdown, no lists, "
    "no emoji, spell out numbers and abbreviations naturally."
)


class Brain:
    def __init__(self, model: str = MODEL) -> None:
        self._model = model
        self._history: list[dict] = []

    def reply(self, user_text: str, timeout: float = 30.0) -> str:
        """Send user text to Ollama, return the full reply text (blocking).

        History is only updated after a successful round-trip, so a failed
        request leaves the conversation state intact.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": user_text},
        ]
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": OLLAMA_OPTIONS,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        message = response.json().get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            raise RuntimeError(f"Ollama returned no content: {response.text[:200]}")
        self._remember_turn(user_text, text)
        return text

    def stream_reply(self, user_text: str, timeout: float = 60.0) -> Iterator[str]:
        """Yield the reply sentence-by-sentence as tokens stream from Ollama.

        Each yielded string is a complete sentence ready for TTS. History is
        updated only after the stream finishes cleanly.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history,
            {"role": "user", "content": user_text},
        ]
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": OLLAMA_OPTIONS,
            },
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()

        buffer = ""
        full_reply: list[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed stream line rather than crash mid-reply
            buffer += (payload.get("message") or {}).get("content") or ""
            sentences, buffer = _split_sentences(buffer)
            for sentence in sentences:
                full_reply.append(sentence)
                yield sentence
            if payload.get("done"):
                break
        tail = buffer.strip()
        if tail:
            full_reply.append(tail)
            yield tail
        if full_reply:
            self._remember_turn(user_text, " ".join(full_reply))

    def _remember_turn(self, user_text: str, reply_text: str) -> None:
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": reply_text})
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-MAX_HISTORY_TURNS * 2 :]

    def warm_up(self) -> None:
        """Load the model into Ollama's memory so the first real reply is fast."""
        requests.post(
            OLLAMA_URL,
            json={
                "model": self._model,
                "messages": [],
                "stream": False,
                "options": OLLAMA_OPTIONS,
            },
            timeout=120,
        )
