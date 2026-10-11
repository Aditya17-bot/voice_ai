"""Read-only desktop actions: open allowlisted apps and sites, browse
allowlisted folders, and read text files so the assistant can explain them.

Safety model:
- No tool writes, deletes, moves, or executes files. There is no "run command".
- Every path is resolved (symlinks and junctions followed) and must stay inside
  one of the allowed roots; sensitive names (.env, keys, .ssh, ...) are never
  listed or read, and text containing a private key is refused.
- Only three sites can be opened. File contents reach the LLM, so a free-form
  URL tool would let a malicious file make the model send data out.
- Programs are launched by absolute path with argument lists, never through a
  shell string or a PATH/current-directory lookup.

Known limitation: checks are by name, so a hard link to a secret under an
innocent name in an allowed folder would be readable. Creating one needs local
write access to that folder.

Tool methods return short plain-English strings that work both as a spoken
confirmation and as a tool result for the LLM. Failures raise ToolError with a
message that is safe to speak aloud.
"""

import difflib
import fnmatch
import os
import re
import stat
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Callable

MAX_READ_BYTES = 4_000  # ~1.2k tokens: fits num_ctx 2048 with the prompt and a short history
README_PREVIEW_BYTES = 2_000
MAX_LIST_ENTRIES = 15  # names spoken per listing; the rest become a count
SUBFOLDER_SEARCH_SECONDS = 0.15  # when a file isn't here, how long to look one level down
MATCH_CUTOFF = 0.75  # fuzzy-match floor for spoken names
EXACT_CUTOFF = 0.98  # name or stem equal after normalizing
PREFIX_SCORE = 0.85  # "voice" -> voice_input
AMBIGUITY_MARGIN = 0.05  # top two this close -> ask instead of guessing

# Spoken name -> (label, URL). The only sites the assistant can open.
SITES = {
    "gmail": ("Gmail", "https://mail.google.com"),
    "youtube": ("YouTube", "https://www.youtube.com"),
    "github": ("GitHub", "https://github.com"),
}

# Spoken name -> canonical app. Keys are compared after _normalize().
APP_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "terminal": "terminal",
    "windows terminal": "terminal",
    "powershell": "terminal",
    "console": "terminal",
    "vs code": "vscode",
    "code": "vscode",
    "visual studio code": "vscode",
    "editor": "vscode",
}
APP_LABELS = {"chrome": "Chrome", "terminal": "the terminal", "vscode": "VS Code"}

# Clutter hidden from listings and fuzzy matching (dotfiles are hidden too).
HIDDEN_NAMES = {"node_modules", "__pycache__", "venv", "desktop.ini", "thumbs.db"}

# Never listed or read, at any depth. Matched case-insensitively per path part.
SENSITIVE_PATTERNS = [
    ".env",
    ".env.*",
    ".ssh",
    ".aws",
    ".gnupg",
    ".netrc",
    ".git-credentials",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "*.ppk",
    "*.jks",
    "*.keystore",
    "*.gpg",
    "*.kdbx",
    "*.token",
    "id_rsa*",
    "id_ed25519*",
    "service-account*",
    "*credential*",
    "*secret*",
    "*password*",
]

_UP_WORDS = {"up", "back", "parent", "goback", "upone", "oneup", "previous"}
_FILLER_WORDS = re.compile(r"\b(?:the|my|folder|directory|file|called|named)\b")
_SPOKEN_DOT = re.compile(r"\s+dot\s+")

# Symlinks, junctions, and other reparse points can lead outside a folder.
# Checked from scandir's cached data, so it costs nothing for plain entries.
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

Launcher = Callable[[list[str], Path | None, bool], None]


class ToolError(Exception):
    """A failed action. str(error) is a friendly sentence safe to speak."""


class NotFound(ToolError):
    """No folder or file matched the spoken name."""


class Ambiguous(ToolError):
    """Two entries matched about equally; the message asks which one."""

    def __init__(self, message: str, options: list[Path]) -> None:
        super().__init__(message)
        self.options = options


class Elsewhere(Ambiguous):
    """Not here, but a subfolder has it; the message offers it."""


def default_roots() -> dict[str, Path]:
    """The folders the assistant may browse: C:\\adi, Desktop, Downloads."""
    home = Path.home()
    # Desktop is often redirected into OneDrive on Windows.
    desktop = next(
        (p for p in (home / "OneDrive" / "Desktop", home / "Desktop") if p.is_dir()),
        home / "Desktop",
    )
    return {"adi": Path("C:/adi"), "desktop": desktop, "downloads": home / "Downloads"}


def _normalize(name: str) -> str:
    """Comparison key: 'Voice Input.' and 'voice_input' both -> 'voiceinput'."""
    name = _SPOKEN_DOT.sub(".", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def _clean_query(query: str) -> str:
    """Drop words like 'the' and 'folder' from a spoken name, if anything's left."""
    stripped = _FILLER_WORDS.sub(" ", query.lower()).strip()
    return stripped or query


def _score(query_key: str, entry: Path) -> float:
    """How well a normalized spoken name matches an entry (1.0 = exact name)."""
    name_key = _normalize(entry.name)
    best = 0.0
    for key in (name_key, _normalize(entry.stem)):
        if key == query_key:
            score = 1.0 if key == name_key else 0.98  # full name beats stem
        elif len(query_key) >= 3 and key.startswith(query_key):
            score = PREFIX_SCORE
        else:
            score = difflib.SequenceMatcher(None, query_key, key).ratio()
        best = max(best, score)
    return best


def _best_match(query: str, entries: list[Path], cutoff: float = MATCH_CUTOFF) -> Path | None:
    """Closest entry by name or stem, None if nothing is close enough.

    Raises ToolError asking the user to choose when two entries score about
    the same — a wrong guess is worse than a quick question.
    """
    q = _normalize(_clean_query(query))
    if not q:
        return None
    scored = sorted(
        ((_score(q, e), e) for e in entries), key=lambda pair: pair[0], reverse=True
    )
    if not scored or scored[0][0] < cutoff:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < AMBIGUITY_MARGIN:
        first, second = scored[0][1], scored[1][1]
        if first.name.lower() == second.name.lower():  # same name, different parents
            raise Ambiguous(
                f"Did you mean {first.name} in {first.parent.name}, "
                f"or in {second.parent.name}?",
                [first, second],
            )
        raise Ambiguous(f"Did you mean {first.name} or {second.name}?", [first, second])
    return scored[0][1]


def _word_runs(utterance: str, longest: int = 4) -> set[str]:
    """Normalized runs of up to `longest` consecutive words: 'voice in put' -> 'voiceinput'."""
    words = [w for w in (_normalize(w) for w in utterance.split()) if w and w != "dot"]
    return {
        "".join(words[i:j])
        for i in range(len(words))
        for j in range(i + 1, min(i + longest, len(words)) + 1)
    }


def named_in(name: str, utterance: str) -> bool:
    """Whole words of the utterance spell the name or its stem: 'the read me' names README.md."""
    runs = _word_runs(utterance)
    keys = {_normalize(name), _normalize(Path(name).stem)}
    return any(len(key) >= 2 and key in runs for key in keys)


def resolve_app(name: str) -> str | None:
    """Canonical app ('chrome', 'terminal', 'vscode') for a spoken name, or None."""
    key = _normalize(_clean_query(name))
    return next((app for alias, app in APP_ALIASES.items() if _normalize(alias) == key), None)


def resolve_site(name: str) -> str | None:
    """SITES key for a spoken name ('G mail', 'youtube.com'), or None."""
    key = _normalize(_clean_query(name))
    key = key.removesuffix("com") if key.removesuffix("com") in SITES else key
    return key if key in SITES else None


def _is_sensitive(name: str) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in SENSITIVE_PATTERNS)


def _is_hidden(name: str) -> bool:
    return name.startswith(".") or name.lower() in HIDDEN_NAMES or _is_sensitive(name)


def _first_file(*candidates: Path | None) -> str | None:
    return next((str(p) for p in candidates if p is not None and p.is_file()), None)


def _env_path(var: str, *parts: str) -> Path | None:
    base = os.environ.get(var)
    return Path(base, *parts) if base else None


def _find_chrome() -> str | None:
    chrome = ("Google", "Chrome", "Application", "chrome.exe")
    return _first_file(
        _env_path("PROGRAMFILES", *chrome),
        _env_path("PROGRAMFILES(X86)", *chrome),
        _env_path("LOCALAPPDATA", *chrome),
    )


def _find_vscode() -> str | None:
    return _first_file(
        _env_path("LOCALAPPDATA", "Programs", "Microsoft VS Code", "Code.exe"),
        _env_path("PROGRAMFILES", "Microsoft VS Code", "Code.exe"),
    )


def _terminal_command(cwd: Path) -> tuple[list[str], bool] | None:
    """argv to open a terminal in cwd and whether it needs a new console."""
    wt = _first_file(_env_path("LOCALAPPDATA", "Microsoft", "WindowsApps", "wt.exe"))
    # wt treats ';' in arguments as a command separator — avoid it for such paths.
    if wt and ";" not in str(cwd):
        return [wt, "-d", str(cwd)], False
    powershell = _first_file(
        _env_path("SYSTEMROOT", "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    )
    return ([powershell, "-NoExit"], True) if powershell else None


def _launch(argv: list[str], cwd: Path | None, new_console: bool) -> None:
    """Start a program detached from our console. Never uses a shell."""
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if new_console else 0
    subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        close_fds=True,
    )


class Workspace:
    """Read-only tools plus the assistant's current folder."""

    def __init__(
        self,
        roots: dict[str, Path] | None = None,
        launcher: Launcher | None = None,
        open_url: Callable[[str], object] | None = None,
    ) -> None:
        resolved = {
            name: path.resolve()
            for name, path in (roots or default_roots()).items()
            if path.is_dir()
        }
        if not resolved:
            raise ValueError("none of the allowed folders exist")
        self._roots = resolved
        self._launch = launcher or _launch
        self._open_url = open_url or webbrowser.open
        self.cwd = next(iter(resolved.values()))

    @property
    def root_names(self) -> list[str]:
        return list(self._roots)

    # ----- path safety -------------------------------------------------

    def _guard(self, path: Path) -> Path:
        """Resolve path and require it inside an allowed root, not hidden or sensitive."""
        resolved = path.resolve()
        for root in self._roots.values():
            if resolved == root or resolved.is_relative_to(root):
                # Hidden too: .git/config, .npmrc and friends can hold tokens.
                if any(_is_hidden(part) for part in resolved.relative_to(root).parts):
                    raise ToolError("That one's private, so I'll leave it alone.")
                return resolved
        raise ToolError("That's outside the folders I'm allowed to look at.")

    def _entries(self, folder: Path) -> tuple[list[Path], list[Path]]:
        """Visible (folders, files) of a folder, sorted by name.

        Links that lead outside the roots or to something private are left
        out entirely, so fuzzy matching never picks them.
        """
        try:
            folder = self._guard(folder)  # re-check: a junction may have been swapped in
            with os.scandir(folder) as it:
                children = sorted(it, key=lambda e: e.name.lower())
        except OSError:
            return [], []
        folders: list[Path] = []
        files: list[Path] = []
        for child in children:
            if _is_hidden(child.name):
                continue
            path = Path(child.path)
            try:
                # Only links can lead outside the guarded folder, so only
                # they need the (slow) resolve — Downloads can hold thousands.
                attributes = getattr(child.stat(follow_symlinks=False), "st_file_attributes", 0)
                if child.is_symlink() or attributes & _REPARSE_POINT:
                    target = self._guard(path)
                    is_dir, is_file = target.is_dir(), target.is_file()
                else:
                    is_dir = child.is_dir(follow_symlinks=False)
                    is_file = child.is_file(follow_symlinks=False)
            except (ToolError, OSError):
                continue
            if is_dir:
                folders.append(path)
            elif is_file:
                files.append(path)
        return folders, files

    def _root_of(self, path: Path) -> Path:
        return next(
            root
            for root in self._roots.values()
            if path == root or path.is_relative_to(root)
        )

    # ----- navigation --------------------------------------------------

    def where(self) -> str:
        """Current folder, spoken as e.g. 'assistant, inside voice_input'."""
        if self.cwd == self._root_of(self.cwd):
            return f"the {self.cwd.name} folder"
        return f"{self.cwd.name}, inside {self.cwd.parent.name}"

    def go_to_folder(self, name: str, exact: bool = False) -> str:
        """Move into a folder by spoken name: a subfolder, a root, or 'up'.

        exact=True skips the fuzzy pass — for loosely phrased requests where a
        near-miss is more likely a misunderstanding than a typo.
        """
        if _normalize(_clean_query(name)) in _UP_WORDS:
            if self.cwd == self._root_of(self.cwd):
                raise ToolError(f"We're already at the top of {self.where()}.")
            self.cwd = self._guard(self.cwd.parent)
            return f"Moved up to {self.where()}."

        # Exact names first — here, then the roots and their direct subfolders
        # (so "go to voice input" works from anywhere) — then fuzzy, but only
        # here, so a near-miss never jumps somewhere unrelated.
        subfolders, _ = self._entries(self.cwd)
        target = _best_match(name, subfolders, cutoff=EXACT_CUTOFF)
        if target is None:
            wider = list(self._roots.values())
            for root in self._roots.values():
                wider.extend(self._entries(root)[0])
            target = _best_match(name, wider, cutoff=EXACT_CUTOFF)
        if target is None and not exact:
            target = _best_match(name, subfolders)
        if target is None:
            raise NotFound(f"I couldn't find a folder called {name}.")
        return self.enter(target)

    def enter(self, folder: Path) -> str:
        """Move into a folder already found (e.g. picked after "did you mean")."""
        target = self._guard(folder)
        if not target.is_dir():
            raise ToolError(f"{folder.name} isn't a folder.")
        self.cwd = target
        return f"Now in {self.where()}."

    def list_folder(self) -> str:
        """Names of the visible folders and files in the current folder."""
        folders, files = self._entries(self.cwd)
        if not folders and not files:
            return f"{self.where().capitalize()} is empty."
        parts = []
        if folders:
            parts.append(f"{_count(folders, 'folder')}: {_join_capped(folders)}")
        if files:
            parts.append(f"{_count(files, 'file')}: {_join_capped(files)}")
        # Separate sentences so speech can start on the first one.
        return f"{self.where().capitalize()} has " + ". It has ".join(parts) + "."

    def names(self, limit: int = 30) -> tuple[list[str], list[str]]:
        """Visible folder and file names here, for the LLM to pick from."""
        folders, files = self._entries(self.cwd)
        return [p.name for p in folders[:limit]], [p.name for p in files[:limit]]

    def explain_folder(self) -> str:
        """Listing plus the start of the README, for the LLM to summarize."""
        summary = self.list_folder()
        _, files = self._entries(self.cwd)
        readme = next((f for f in files if f.stem.lower() == "readme"), None)
        if readme is None:
            return summary
        try:
            text, _ = self._read_text(self._guard(readme), README_PREVIEW_BYTES)
        except ToolError:
            return summary
        return f"{summary}\n\nStart of {readme.name}:\n{text}"

    def read_file(self, name: str, exact: bool = False) -> str:
        """Text of a file in the current folder (first MAX_READ_BYTES)."""
        _, files = self._entries(self.cwd)
        target = _best_match(name, files, cutoff=EXACT_CUTOFF if exact else MATCH_CUTOFF)
        if target is None:
            nearby = self._files_one_level_down(name)
            if len(nearby) == 1:
                found = nearby[0]
                raise Elsewhere(
                    f"{found.name} isn't here, but there's one in {found.parent.name}. "
                    "Want that one?",
                    nearby,
                )
            if nearby:
                raise Elsewhere(
                    f"{nearby[0].name} isn't here. Did you mean the one in "
                    f"{nearby[0].parent.name}, or in {nearby[1].parent.name}?",
                    nearby[:2],
                )
            raise NotFound(f"I couldn't find a file called {name} here.")
        return self.read(target)

    def _files_one_level_down(self, name: str) -> list[Path]:
        """Exact-name matches in the subfolders here ("main.py" asked for in the project root).

        Plain name comparison, no fuzzy scoring, within a time budget: a miss
        in a crowded folder must not stall the reply.
        """
        key = _normalize(_clean_query(name))
        if len(key) < 2:
            return []
        deadline = time.perf_counter() + SUBFOLDER_SEARCH_SECONDS
        found = []
        for sub in self._entries(self.cwd)[0]:
            if time.perf_counter() > deadline:
                break
            try:
                with os.scandir(sub) as it:
                    names = [entry.name for entry in it]
            except OSError:
                continue
            hits = [
                n for n in names
                if not _is_hidden(n) and key in (_normalize(n), _normalize(Path(n).stem))
            ]
            if len(hits) != 1:
                continue  # none, or unclear even within one folder: don't offer it
            try:
                path = self._guard(sub / hits[0])
            except ToolError:
                continue
            if path.is_file():
                found.append(path)
        return found

    def read(self, file: Path) -> str:
        """Text of a file already found (e.g. picked after "did you mean")."""
        target = self._guard(file)
        if not target.is_file():
            raise ToolError(f"{file.name} isn't a file.")
        text, truncated = self._read_text(target, MAX_READ_BYTES)
        note = " (only the beginning — the file is longer)" if truncated else ""
        return f"Contents of {target.name}{note}:\n{text}"

    def _read_text(self, path: Path, limit: int) -> tuple[str, bool]:
        try:
            with path.open("rb") as f:
                data = f.read(limit + 1)
        except OSError:
            raise ToolError(f"I couldn't open {path.name}.") from None
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16, e.g. PowerShell `>`
            text = data[:limit].decode("utf-16", errors="replace")
        elif b"\0" in data[:4096]:
            raise ToolError(f"{path.name} isn't a text file, so I can't read it out.")
        else:
            text = data[:limit].decode("utf-8", errors="replace")
        if "PRIVATE KEY-----" in text:
            raise ToolError(f"{path.name} holds a private key, so I'll leave it alone.")
        return text, len(data) > limit

    # ----- launching ---------------------------------------------------

    def open_app(self, name: str) -> str:
        """Open Chrome, a terminal, or VS Code."""
        app = self._resolve_app(name)
        if app == "chrome":
            self._launch([self._require(_find_chrome(), "Chrome")], None, False)
        elif app == "terminal":
            self._open_terminal(Path.home())
        else:
            self._launch([self._require(_find_vscode(), "VS Code")], None, False)
        return f"Opening {APP_LABELS[app]}."

    def open_here(self, name: str) -> str:
        """Open a terminal or VS Code in the current folder."""
        app = self._resolve_app(name)
        if app == "terminal":
            self._open_terminal(self.cwd)
        elif app == "vscode":
            self._launch([self._require(_find_vscode(), "VS Code"), str(self.cwd)], None, False)
        else:
            raise ToolError("I can only open the terminal or VS Code in a folder.")
        return f"Opening {APP_LABELS[app]} in {self.where()}."

    def open_website(self, site: str) -> str:
        """Open Gmail, YouTube, or GitHub — nothing else."""
        key = resolve_site(site)
        if key is None:
            raise ToolError("I can only open Gmail, YouTube, or GitHub.")
        label, url = SITES[key]
        chrome = _find_chrome()
        if chrome:
            self._launch([chrome, url], None, False)
        else:
            self._open_url(url)
        return f"Opening {label}."

    def _open_terminal(self, cwd: Path) -> None:
        command = _terminal_command(cwd)
        if command is None:
            raise ToolError("I can't find a terminal on this computer.")
        argv, new_console = command
        self._launch(argv, cwd, new_console)

    def _resolve_app(self, name: str) -> str:
        app = resolve_app(name)
        if app is not None:
            return app
        raise ToolError(f"I can't open {name}. I can open Chrome, the terminal, or VS Code.")

    @staticmethod
    def _require(executable: str | None, label: str) -> str:
        if executable is None:
            raise ToolError(f"I can't find {label} on this computer.")
        return executable


def _count(items: list[Path], noun: str) -> str:
    return f"{len(items)} {noun}" + ("" if len(items) == 1 else "s")


def _join_capped(paths: list[Path]) -> str:
    names = [p.name for p in paths[:MAX_LIST_ENTRIES]]
    extra = len(paths) - len(names)
    return ", ".join(names) + (f", and {extra} more" if extra > 0 else "")


if __name__ == "__main__":
    # Manual check without voice: ls | cd <name> | cat <name> | explain |
    # open <app> | here <app> | site <name> | where
    import sys

    if hasattr(sys.stdout, "reconfigure"):  # file text may hold emoji; cp1252 can't print them
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ws = Workspace()
    commands = {
        "ls": lambda _: ws.list_folder(),
        "explain": lambda _: ws.explain_folder(),
        "where": lambda _: ws.where(),
        "cd": ws.go_to_folder,
        "cat": ws.read_file,
        "open": ws.open_app,
        "here": ws.open_here,
        "site": ws.open_website,
    }
    while True:
        try:
            verb, _, arg = input("tools> ").strip().partition(" ")
        except (EOFError, KeyboardInterrupt):
            break
        if verb not in commands:
            print("commands:", ", ".join(commands))
            continue
        try:
            print(commands[verb](arg))
        except ToolError as exc:
            print(f"[refused] {exc}")
