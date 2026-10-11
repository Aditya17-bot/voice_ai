"""Measure the LLM classifier fallback on requests the fast-path router misses.

Not a unit test: needs Ollama running. Each utterance runs through the real
pipeline (router, gate, classifier, checks, tools) in a throwaway folder
tree; nothing is launched. TUNING is the set the checks were tuned against;
HELD_OUT was written afterwards and is the honest score.

Run: python -m tests.eval_classifier [model ...]
"""

import sys
import tempfile
import time
from pathlib import Path

from assistant.brain import Brain
from assistant.classifier import classify_request
from assistant.router import Intent, execute, might_be_command, route
from assistant.tools import ToolError, Workspace, _normalize

Expected = tuple[str, str] | list[tuple[str, str]] | None
# (utterance, folder to start in, expected (action, arg), alternatives, or None for "just talk")
Case = tuple[str, str, Expected]

TUNING: list[Case] = [
    ("Pop open a browser for me.", "adi", ("open_app", "chrome")),
    ("I need to check my mail.", "adi", ("open_website", "gmail")),
    ("Fire up the code editor.", "adi", [("open_app", "vscode"), ("open_here", "vscode")]),
    ("Put on some YouTube.", "adi", ("open_website", "youtube")),
    ("Open the GitHub website.", "adi", ("open_website", "github")),
    ("What's this repo for?", "adi/voice_input", ("explain_folder", "")),
    ("Hop into the assistant folder.", "adi/voice_input", ("go_to_folder", "assistant")),
    ("Let's look at the tests.", "adi/voice_input", ("go_to_folder", "tests")),
    ("Get me into the voice input project.", "adi", ("go_to_folder", "voice_input")),
    ("Go to voice in put.", "adi", ("go_to_folder", "voice_input")),
    ("Take a look at my desktop.", "adi", ("go_to_folder", "desktop")),
    ("Go one level up.", "adi/voice_input/assistant", ("go_to_folder", "up")),
    ("What files do we have here?", "adi/voice_input", ("list_folder", "")),
    ("Which folder is this?", "adi/voice_input", ("where", "")),
    ("Tell me about the requirements file.", "adi/voice_input", ("read_file", "requirements.txt")),
    ("Show me what the router does.", "adi/voice_input/assistant", ("read_file", "router.py")),
    ("Open up a command line in this project.", "adi/voice_input", ("open_here", "terminal")),
    ("Can you open this project in the editor?", "adi/voice_input", ("open_here", "vscode")),
    ("Go to sleep.", "adi", None),
    ("How do I open a file in Python?", "adi", None),
    ("Explain recursion.", "adi", None),
    ("What's the best code editor?", "adi", None),
    ("Can you read me a poem?", "adi", None),
    ("Where is Paris?", "adi", None),
    ("Show me how to make pancakes.", "adi", None),
    ("Let's start over.", "adi", None),
    ("What does a terminal do?", "adi", None),
    ("Tell me about the history of YouTube.", "adi", None),
    ("Should I check my email before bed?", "adi", None),
]

HELD_OUT: list[Case] = [
    ("Could you bring up Chrome real quick?", "adi", ("open_app", "chrome")),
    ("I wanna see my inbox.", "adi", ("open_website", "gmail")),
    ("Load up GitHub.", "adi", ("open_website", "github")),
    ("Read me the architecture doc.", "adi/voice_input", ("read_file", "ARCHITECTURE.md")),
    ("Head back up a level.", "adi/voice_input/assistant", ("go_to_folder", "up")),
    ("Can you tell me what this folder is about?", "adi/voice_input", ("explain_folder", "")),
    ("Show me what's inside the assistant folder.", "adi/voice_input", [("go_to_folder", "assistant"), ("list_folder", "")]),
    ("Get the terminal going in this folder.", "adi/voice_input", ("open_here", "terminal")),
    ("Read the zebra notes.", "downloads", [("read_file", "zebra_notes.txt"), ("read_file", "zebra notes")]),
    ("Don't open Chrome.", "adi", None),
    ("Can you check my inbox later, not now?", "adi", None),
    ("I see what you mean.", "adi", None),
    ("Show me the code for the login page.", "adi", None),
    ("Explain this to me again.", "adi", None),
    ("Start a story about a shell on the beach.", "adi", None),
    ("Let me see.", "adi", None),
    ("What's on YouTube these days?", "adi", None),
    ("My project at work is stressful.", "adi/voice_input", None),
    ("I left my files at home.", "adi/voice_input", None),
    ("The terminal at the airport was crowded.", "adi", None),
]

TREE = {
    "adi/voice_input/assistant": ["main.py", "stt.py", "tts.py", "router.py", "brain.py"],
    "adi/voice_input/tests": ["test_tools.py"],
    "adi/voice_input": ["README.md", "requirements.txt", "ARCHITECTURE.md"],
    "adi/school_2019": [],
    "desktop": ["notes.txt"],
    # A crowded Downloads: the name budget must still show the one asked for.
    "downloads": [f"invoice_{i:03d}_scan_from_the_office.pdf" for i in range(300)] + ["zebra_notes.txt"],
}


def _build_tree(base: Path) -> dict[str, Path]:
    for folder, files in TREE.items():
        (base / folder).mkdir(parents=True, exist_ok=True)
        for name in files:
            (base / folder / name).write_text(f"# {name}\nPlaceholder.\n", encoding="utf-8")
    return {name: base / name for name in ("adi", "desktop", "downloads")}


def _matches(intent: Intent | None, expected: Expected) -> bool:
    if expected is None or intent is None:
        return intent is None and expected is None
    options = expected if isinstance(expected, list) else [expected]
    return any(
        intent.action == action and _normalize(intent.arg) == _normalize(arg)
        for action, arg in options
    )


def run_case(brain: Brain, ws: Workspace, text: str) -> tuple[str, Intent | None, float]:
    """(path taken, intent that acted or None, ms to the decision)."""
    started = time.perf_counter()
    intent = route(text)
    if intent is not None:
        try:
            if execute(intent, ws) is not None:
                return "router", intent, (time.perf_counter() - started) * 1000
        except ToolError:
            return "router-refused", None, (time.perf_counter() - started) * 1000
    if not might_be_command(text):
        return "chat", None, 0.0
    intent = classify_request(brain, ws, text)
    ms = (time.perf_counter() - started) * 1000
    if intent is None:
        return "llm: just talk", None, ms
    try:
        if execute(intent, ws) is None:
            return f"llm: {intent.action}({intent.arg}) not found", None, ms
    except ToolError as exc:
        return f"llm: {intent.action}({intent.arg}) refused: {exc}", None, ms
    return f"llm: {intent.action}({intent.arg})", intent, ms


def score(brain: Brain, name: str, cases: list[Case], tmp: Path, roots: dict[str, Path]) -> None:
    right = wrong_actions = 0
    llm_ms: list[float] = []
    print(f"\n--- {name} ---")
    for text, start, expected in cases:
        ws = Workspace(roots=roots, launcher=lambda *a: None, open_url=lambda u: None)
        ws.cwd = (tmp / start).resolve()
        path, intent, ms = run_case(brain, ws, text)
        ok = _matches(intent, expected)
        right += ok
        wrong_actions += intent is not None and not ok
        if path.startswith("llm"):
            llm_ms.append(ms)
        print(f"{'ok ' if ok else 'BAD'} {ms:6.0f} ms  {text:46} {path}")
    llm_ms.sort()
    median = f"{llm_ms[len(llm_ms) // 2]:.0f} ms" if llm_ms else "n/a"
    print(f"{name}: {right}/{len(cases)} right, {wrong_actions} wrong actions; classifier median {median}")


def evaluate(model: str) -> None:
    brain = Brain(model)
    brain.warm_up()
    print(f"\n=== {model} ===")
    with tempfile.TemporaryDirectory() as tmp:
        roots = _build_tree(Path(tmp))
        score(brain, "tuning", TUNING, Path(tmp), roots)
        score(brain, "held out", HELD_OUT, Path(tmp), roots)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name in sys.argv[1:] or ["llama3.2:3b"]:
        evaluate(name)
