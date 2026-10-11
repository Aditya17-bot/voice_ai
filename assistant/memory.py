"""Facts the user asked the assistant to remember ("remember that my exam is on Friday").

This is the one thing the assistant writes, and only to its own file
(data/memory.json in the project, ignored by git), never to the user's
files. Facts are always the user's own words, taken by the router from what
they said: never text from the LLM, a file, an email, or a web page.
"""

import difflib
import json
import re
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.json"
MAX_FACTS = 40
MAX_FACT_CHARS = 200
FORGET_CUTOFF = 0.45  # how close "forget my exam" must be to a stored fact
NOTE_MAX_CHARS = 600  # of facts given to the chat model each turn
_VAGUE = {"me", "my", "it", "that", "this", "about", "the", "a", "i", "you", "your", "thing", "stuff"}

# Turn the user's words into the assistant's: "my exam" -> "your exam".
_PRONOUNS = [
    (r"\bi am\b", "you are"), (r"\bi'm\b", "you're"), (r"\bi've\b", "you've"), (r"\bi'll\b", "you'll"),
    (r"\bi was\b", "you were"), (r"\bmyself\b", "yourself"), (r"\bmy\b", "your"), (r"\bmine\b", "yours"),
    (r"\bme\b", "you"), (r"\bi\b", "you"),
]


def as_said_back(fact: str) -> str:
    """'my exam is on friday' -> 'your exam is on friday'."""
    for pattern, replacement in _PRONOUNS:
        fact = re.sub(pattern, replacement, fact, flags=re.IGNORECASE)
    return fact


class Memory:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self._path = path
        self.facts: list[str] = self._read()

    def _read(self) -> list[str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [str(f["text"])[:MAX_FACT_CHARS] for f in data.get("facts", []) if isinstance(f, dict)][:MAX_FACTS]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return []

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_suffix(".tmp")
        payload = {"facts": [{"text": f, "saved": time.strftime("%Y-%m-%d")} for f in self.facts]}
        temp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        temp.replace(self._path)  # never a half-written file

    def remember(self, fact: str) -> str:
        fact = " ".join(fact.split())[:MAX_FACT_CHARS].rstrip(" .")
        if not fact:
            return "What should I remember?"
        if fact.lower() in (f.lower() for f in self.facts):
            return "I already remember that."
        if len(self.facts) >= MAX_FACTS:
            return f"My memory is full at {MAX_FACTS} things. Ask me to forget something first."
        self.facts.append(fact)
        self._write()
        return f"Got it. I'll remember {as_said_back(fact)}."

    def recall(self) -> str:
        if not self.facts:
            return "You haven't asked me to remember anything yet."
        listed = " ".join(f"{as_said_back(f).capitalize()}." for f in self.facts[-8:])
        more = f" And {len(self.facts) - 8} older things." if len(self.facts) > 8 else ""
        return f"Here's what I remember. {listed}{more}"

    def find(self, about: str) -> str | None:
        """The stored fact "forget about X" most likely means, or None."""
        key = " ".join(w for w in about.lower().split() if w not in _VAGUE)
        if len(key) < 3 or not self.facts:
            return None  # "forget me", "forget about it": too vague to pick one
        scored = [
            (difflib.SequenceMatcher(None, key, f.lower()).ratio() + (0.5 if key in f.lower() else 0), f)
            for f in self.facts
        ]
        score, fact = max(scored)
        return fact if score >= FORGET_CUTOFF else None

    def forget(self, fact: str) -> str:
        """Remove a stored fact (as found by find())."""
        if fact not in self.facts:
            return "I'd already forgotten that."
        self.facts.remove(fact)
        self._write()
        return f"Okay, I've forgotten that {as_said_back(fact)}."

    def forget_all(self) -> str:
        count = len(self.facts)
        self.facts = []
        self._write()
        return f"Done. I've forgotten all {count} things." if count else "My memory was already empty."

    def prompt_note(self) -> str:
        """For the chat model: what the user asked it to keep in mind (newest
        first, bounded so a file or email still fits in the context)."""
        kept, size = [], 0
        for fact in reversed(self.facts):
            if size + len(fact) > NOTE_MAX_CHARS:
                break
            kept.append(fact)
            size += len(fact) + 2
        if not kept:
            return ""
        return "Things I asked you to remember about me: " + "; ".join(kept) + "."
