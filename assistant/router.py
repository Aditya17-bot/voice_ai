"""Fast path: map common spoken commands straight to tools, no LLM decision.

A wrong action is worse than a miss, so phrasing decides how hard we try:
- Explicit commands ("cd tests", "go to the tests folder", "read main.py")
  fuzzy-match names and, if nothing matches, say so.
- Loose phrasing ("go to X", "explain X", "open X") acts only on an exact
  name match; otherwise the utterance falls through to the LLM, so "go to
  sleep" or "explain node.js" stay conversation.
Questions and chit-chat that match no pattern go straight to the LLM.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .tools import (
    Elsewhere,
    NotFound,
    ToolError,
    Workspace,
    _normalize,
    resolve_app,
    resolve_site,
)


@dataclass(frozen=True)
class Intent:
    action: str  # a key of _ACTIONS / _EXPLAIN_ACTIONS, or "open_any"
    arg: str = ""
    loose: bool = False  # exact names only; on a miss, hand over to the LLM


@dataclass(frozen=True)
class Outcome:
    text: str
    explain: bool  # True: the LLM should talk about `text`; False: speak it as-is


# Punctuation Whisper adds. Dots inside names ("main.py") are kept.
_PUNCTUATION = re.compile(r"[,;:!?\"]|\.(?=\s|$)")
_POLITE_PREFIX = re.compile(
    r"^(?:hey|hi|okay|ok|so|um|uh|please|now|just|let's|go ahead and|"
    r"can you|could you|would you|will you|"
    r"i want you to|i'd like you to|i would like you to)\s+"
)
_POLITE_SUFFIX = re.compile(r"\s+(?:please|for me|now|thanks|thank you)$")
_ARTICLE = re.compile(r"^(?:the|a|an|my)\s+")
# A file name inside a phrase: "main.py", "main dot py in simple terms".
_FILE_NAME = re.compile(r"^.*?(?:\.| dot )[a-z0-9]{1,5}\b")
_EMAIL_WORDS = {"email", "emails", "mail", "inbox", "my email", "my inbox", "my mail"}

_HERE = r"(?:here|in here|in (?:this|the current) (?:folder|directory|project))"
_OPEN = r"(?P<verb>open|launch|start|bring up|pull up|fire up)(?: up)?"


def _prepare(text: str) -> str:
    """Lowercase, drop punctuation and politeness: 'Okay, open Chrome please.' -> 'open chrome'."""
    s = " ".join(_PUNCTUATION.sub(" ", text.lower()).split())
    while True:
        stripped = _POLITE_SUFFIX.sub("", _POLITE_PREFIX.sub("", s))
        if stripped == s:
            return s
        s = stripped


def _target(raw: str) -> str:
    return _ARTICLE.sub("", raw.strip())


def _open_here(m: re.Match) -> Intent | None:
    app = resolve_app(_target(m["app"]))
    return Intent("open_here", app) if app in ("terminal", "vscode") else None


# "Explain it again", "what does that do": about the last file, not a file named "it".
_PRONOUN_TARGET = re.compile(r"(?:it|that|this|them|those|these)\b.*")


def _read(m: re.Match) -> Intent | None:
    name = _target(m["name"])
    if _PRONOUN_TARGET.fullmatch(name):
        return None
    file_name = _FILE_NAME.match(name)
    if file_name:
        name = file_name.group(0)  # drop trailing words: "main.py in simple terms"
    # Only "read <file.ext>" is explicit; "explain node.js" or "describe the
    # file system" are just as likely general questions.
    explicit = m["verb"] == "read" and file_name is not None
    return Intent("read_file", name, loose=not explicit)


def _go(m: re.Match) -> Intent:
    # The rule has three alternatives ("go to X", "change folder to X", "cd X").
    name = _target(next(m[g] for g in ("name", "name2", "name3") if m[g]))
    if resolve_site(name):  # "go to gmail"
        return Intent("open_website", name)
    explicit = m["name"] is None or name.endswith((" folder", " directory"))
    return Intent("go_to_folder", name, loose=not explicit)


# "open the github website", "open the chrome app" name the same thing.
_KIND_SUFFIX = re.compile(r" (?:website|web site|site|page|app|application|program)$")


def _open(m: re.Match) -> Intent | None:
    target = _KIND_SUFFIX.sub("", _target(m["target"]))
    if resolve_app(target):
        return Intent("open_app", target)
    if resolve_site(target):
        return Intent("open_website", target)
    if target in _EMAIL_WORDS:
        return Intent("open_website", "gmail")
    if target.endswith((" folder", " directory")):
        return Intent("go_to_folder", target)
    if _FILE_NAME.match(target) or target.endswith(" file"):
        return Intent("read_file", target)
    # "open spotify" deserves the menu of what we can open; "start over" or
    # "open up about your day" are conversation.
    explicit = m["verb"] in ("open", "launch") and len(target.split()) <= 2
    return Intent("open_any", target, loose=not explicit)


# Tried in order; each regex must match the whole prepared utterance. A builder
# may return None to let later rules (and finally the LLM) have a go.
_RULES: list[tuple[str, Callable[[re.Match], Intent | None]]] = [
    (
        r"where am i|what folder am i in|which folder am i in"
        r"|what(?:'s| is) the current folder",
        lambda m: Intent("where"),
    ),
    (
        r"(?:go|move|navigate) (?:up|back)(?: one)?(?: (?:a )?(?:folder|level|directory))?"
        r"|go to (?:the )?parent(?: folder| directory)?",
        lambda m: Intent("go_to_folder", "up"),
    ),
    (
        r"what(?:'s| is| are)? (?:in|inside) (?:here|this folder|this directory|the current folder)"
        r"|what(?:'s| is) here|what files are (?:here|in here|in this folder)"
        rf"|(?:list|show)(?: me)? (?:the |all )?(?:the )?(?:files|folders|contents)(?: {_HERE})?"
        r"|list (?:this|the current) folder",
        lambda m: Intent("list_folder"),
    ),
    (
        # Trailing words are fine: "explain this project in one sentence".
        r"(?:explain|describe|summari[sz]e|tell me about|what(?:'s| is)) "
        r"(?:this|the current) (?:folder|project|directory|repo)\b.*",
        lambda m: Intent("explain_folder"),
    ),
    (rf"{_OPEN} (?P<app>.+?) {_HERE}", _open_here),
    (r"open (?:this|the current) (?:folder|project|directory) in (?P<app>.+)", _open_here),
    (
        r"(?P<verb>read|explain|describe|summari[sz]e|show me|what(?:'s| is) in)(?: me)? "
        r"(?:the )?(?:file )?(?P<name>.+?)(?: file)?",
        _read,
    ),
    (r"what does (?P<verb>)(?P<name>.+?) do", _read),
    (
        r"(?:go|navigate|move|switch|jump|take me|head)(?: over| back)? (?:in)?to (?P<name>.+)"
        r"|change (?:the )?(?:folder|directory) to (?P<name2>.+)|cd (?P<name3>.+)",
        _go,
    ),
    (rf"{_OPEN} (?P<target>.+)", _open),
]
_COMPILED = [(re.compile(pattern), build) for pattern, build in _RULES]


# Words that suggest a desktop request the rules above didn't catch ("pop open
# a browser", "what's this repo for"). Only these utterances go to the LLM
# classifier; chit-chat skips it, which is faster and can't misfire.
_COMMAND_HINT = re.compile(
    r"\b(?:open|launch|start|show|look|check|go|navigate|move|switch|jump"
    r"|take|head|cd|folders?|director(?:y|ies)|files?|read|explain|describe"
    r"|summari[sz]e|list|browser|chrome|terminal|console|powershell|vs code|vscode"
    r"|editor|gmail|e?mails?|inbox|youtube|github|project|repo|here|inside|where"
    r"|readme|downloads|desktop|pull up|bring up)\b"
    r"|\w\.[a-z0-9]{1,5}\b|\bdot [a-z0-9]{1,5}\b"
)


# Advice and general-knowledge questions stay conversation even when they name
# a command word: "how do I open a file in Python", "should I check my email".
_QUESTION = re.compile(
    r"^(?:how (?:do|can|would|should|does)|should|why|is it|is there|do you"
    r"|where (?:is|are|do|does|can|should)"
    r"|what(?:'s| is| are) (?:a|an|the best|your)|what does an? )\b"
)


def might_be_command(text: str) -> bool:
    """True if an utterance the rules missed could still be a desktop request."""
    prepared = _prepare(text)
    return not _QUESTION.match(prepared) and bool(_COMMAND_HINT.search(prepared))


def route(text: str) -> Intent | None:
    """Intent for a fast-path command, or None to hand the utterance to the LLM."""
    prepared = _prepare(text)
    if not prepared:
        return None
    for pattern, build in _COMPILED:
        match = pattern.fullmatch(prepared)
        if match:
            intent = build(match)
            if intent is not None:
                return intent
    return None


# Actions whose result is spoken directly, and how to call them.
_ACTIONS: dict[str, Callable[[Workspace, Intent], str]] = {
    "where": lambda ws, i: f"We're in {ws.where()}.",
    "list_folder": lambda ws, i: ws.list_folder(),
    "go_to_folder": lambda ws, i: ws.go_to_folder(i.arg, exact=i.loose),
    "open_app": lambda ws, i: ws.open_app(i.arg),
    "open_here": lambda ws, i: ws.open_here(i.arg),
    "open_website": lambda ws, i: ws.open_website(i.arg),
}
# Actions whose result is material for the LLM to explain.
_EXPLAIN_ACTIONS: dict[str, Callable[[Workspace, Intent], str]] = {
    "explain_folder": lambda ws, i: ws.explain_folder(),
    "read_file": lambda ws, i: ws.read_file(i.arg, exact=i.loose),
}


def execute(intent: Intent, workspace: Workspace) -> Outcome | None:
    """Run an intent. None means "not ours after all — ask the LLM".

    Raises ToolError with a speakable message for explicit commands that fail.
    """
    try:
        return _run(intent, workspace)
    except NotFound:
        if intent.loose:
            return None
        raise
    except Elsewhere:
        # "What does docker compose do" is a question, not a file request; a
        # loose phrasing earns the subfolder offer only with an extension.
        if intent.loose and not _FILE_NAME.match(intent.arg):
            return None
        raise


def _run(intent: Intent, workspace: Workspace) -> Outcome:
    if intent.action in _ACTIONS:
        return Outcome(_ACTIONS[intent.action](workspace, intent), explain=False)
    if intent.action in _EXPLAIN_ACTIONS:
        return Outcome(_EXPLAIN_ACTIONS[intent.action](workspace, intent), explain=True)
    if intent.action == "open_any":
        # "open voice input" -> folder; "open readme" -> file; exact names only.
        try:
            return Outcome(workspace.go_to_folder(intent.arg, exact=True), explain=False)
        except NotFound:
            pass
        try:
            return Outcome(workspace.read_file(intent.arg, exact=True), explain=True)
        except NotFound:
            raise NotFound(
                f"I can't open {intent.arg}. I can open Chrome, the terminal, "
                "VS Code, Gmail, YouTube, GitHub, or one of your folders."
            ) from None
    raise ValueError(f"unknown action {intent.action!r}")


# ----- follow-ups ---------------------------------------------------------

_ORDINALS = {"first": 0, "1st": 0, "former": 0, "second": 1, "2nd": 1, "latter": 1, "other": 1, "last": 1}
_YES = {"yes", "yeah", "yep", "sure", "ok", "okay"}
_NO = {"no", "not", "nope", "nah", "don't", "doesn't", "wrong", "neither", "never"}
# Words an answer may carry around the choice itself: "yes, the one in work".
_ANSWER_FILLER = _YES | {"the", "one", "in", "i", "meant", "mean", "that", "please"}
MAX_ANSWER_WORDS = 6


def pick_option(text: str, options: list[Path]) -> Path | None:
    """The option an answer to "did you mean X or Y?" (or "want that one?") picks, or None.

    Only short answers count: "yes", "the first one", "the other", the
    option's name ("voice input"), or, when both share a name, its folder
    ("the one in work"). Anything else ("go to downloads", "the first time I
    ran it") returns None and is handled as a fresh request.
    """
    words = _prepare(text).split()
    if not words or len(words) > MAX_ANSWER_WORDS or _NO & set(words):
        return None
    rest = [w for w in words if w not in _ANSWER_FILLER]
    if not rest:
        return options[0] if len(options) == 1 else None  # "yes", "that one"
    if len(rest) == 1 and rest[0] in _ORDINALS and _ORDINALS[rest[0]] < len(options):
        return options[_ORDINALS[rest[0]]]
    key = _normalize("".join(rest))
    same_name = len({o.name.lower() for o in options}) < len(options)
    labels = [o.parent.name if same_name else o.name for o in options]
    named = [o for o, label in zip(options, labels) if key in (_normalize(label), _normalize(Path(label).stem))]
    return named[0] if len(named) == 1 else None


# A question about the file just explained: "what else does it do?", "tell me
# more about the code". Anchored to the file on purpose; "what time is it"
# must stay small talk.
_REFERS_BACK = re.compile(
    r"\b(?:what else|tell me more|explain more|go on|keep going)\b"
    r"|\b(?:it|that|this) (?:do|does|say|says|mean|means|contain|contains|for)\b"
    r"|\b(?:does|is|was) (?:it|that|this) (?:do|say|for|used|work|written)\b"
    r"|\b(?:the|that|this) (?:file|code|function|script)\b"
    r"|\b(?:about|in|explain) (?:it|that) again\b|\bmore about (?:it|that)\b"
)


def refers_back(text: str) -> bool:
    """A question about the last file — and not naming a different one."""
    prepared = _prepare(text)
    return bool(_REFERS_BACK.search(prepared)) and not _FILE_NAME.match(prepared)


def open_path(workspace: Workspace, path: Path) -> Outcome:
    """Enter a folder or read a file that was picked from a "did you mean" list."""
    if path.is_dir():
        return Outcome(workspace.enter(path), explain=False)
    return Outcome(workspace.read(path), explain=True)
