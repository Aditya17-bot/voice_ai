"""Read-only GitHub: your repos, what's new on one, its open issues and pull
requests, stars, and what a repo is about (its README, explained by the LLM).

Uses a fine-grained token created with read-only permissions, so GitHub
itself refuses any change. Set it up once with `python -m assistant.setup
github`; it's kept encrypted (see credentials.py). Only GET requests are sent.

Repo text (README, titles, commit messages) is data: spoken directly or sent
to the LLM as delimited data, never to the classifier that chooses actions.
"""

import base64
import difflib
import re
import time
from datetime import datetime, timezone
from typing import Callable

import requests

from . import credentials

API = "https://api.github.com"
TIMEOUT_S = 8
SECRET_NAME = "github"
LISTED = 5
MAX_README_CHARS = 3000  # with the prompt, fits the LLM's 2048-token context
REPO_CACHE_S = 600
MATCH_CUTOFF = 0.6
_SPEAKABLE = re.compile(r"[^\w\s.,:;!?'&()/#@-]")


class GitHubError(Exception):
    """Speakable reason a GitHub request didn't work."""


class NotARepo(GitHubError):
    """No repo of yours by that name: "issues on my mind" wasn't about GitHub."""


def clean(text: str, limit: int = 90) -> str:
    text = "".join(ch for ch in str(text) if ord(ch) <= 0xFFFF)
    text = " ".join(_SPEAKABLE.sub(" ", text).split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _spoken_name(repo: str) -> str:
    return repo.replace("_", " ").replace("-", " ")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _ago(stamp: str) -> str:
    try:
        then = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return ""
    days = (datetime.now(timezone.utc) - then).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    return then.strftime("in %B %Y")


class GitHub:
    def __init__(
        self,
        load_login: Callable[[], dict | None] = lambda: credentials.load(SECRET_NAME),
        http: requests.Session | None = None,
    ) -> None:
        self._load_login = load_login
        self._http = http or requests.Session()
        self._repos: list[dict] = []
        self._repos_at = 0.0

    @property
    def connected(self) -> bool:
        return self._load_login() is not None

    def _get(self, path: str, **params) -> dict | list:
        """GET only: this class never sends a request that could change anything."""
        login = self._load_login()
        if login is None:
            raise GitHubError("GitHub isn't connected yet. Run the GitHub setup once, it's in the read me.")
        try:
            response = self._http.get(
                f"{API}/{path}", params=params, timeout=TIMEOUT_S,
                headers={"Authorization": f"Bearer {login['token']}", "Accept": "application/vnd.github+json",
                         "X-GitHub-Api-Version": "2022-11-28"},
            )
        except requests.RequestException:
            raise GitHubError("I couldn't reach GitHub just now.") from None
        if response.status_code == 401:
            raise GitHubError("GitHub didn't accept the token. It may have expired: run the GitHub setup again.")
        if response.status_code in (403, 404):
            raise GitHubError("GitHub wouldn't show me that. The token may not cover that repo.")
        if not response.ok:
            raise GitHubError("GitHub didn't answer that one.")
        return response.json()

    def _my_repos(self) -> list[dict]:
        if not self._repos or time.time() - self._repos_at > REPO_CACHE_S:
            self._repos = list(self._get("user/repos", sort="pushed", per_page=100, affiliation="owner"))
            self._repos_at = time.time()
        return self._repos

    def _repo(self, spoken: str) -> dict:
        """One of your repos by spoken name: 'voice ai' -> voice_ai."""
        repos = self._my_repos()
        key = _key(re.sub(r"^(?:the|my) ", "", spoken.strip().lower()))
        exact = [r for r in repos if _key(r["name"]) == key]
        if exact:
            return exact[0]
        names = {_key(r["name"]): r for r in repos}
        close = difflib.get_close_matches(key, names, n=2, cutoff=MATCH_CUTOFF)
        if close and difflib.SequenceMatcher(None, key, close[0]).ratio() > 0.85:
            return names[close[0]]  # "voice ay" -> voiceai
        if close:  # not sure: ask rather than show the wrong repo
            options = " or ".join(_spoken_name(names[c]["name"]) for c in close)
            raise GitHubError(f"Did you mean {options}?")
        raise NotARepo(f"I don't see a repo of yours called {clean(spoken, 40)}.")

    # ----- what the assistant can say -------------------------------------

    def repos(self) -> str:
        repos = self._my_repos()
        if not repos:
            return "You don't have any repos yet."
        recent = ", ".join(_spoken_name(r["name"]) for r in repos[:LISTED])
        return f"You have {len(repos)} repos. The most recently updated are {recent}."

    def whats_new(self, spoken: str) -> str:
        repo = self._repo(spoken)
        commits = self._get(f"repos/{repo['full_name']}/commits", per_page=3)
        if not commits:
            return f"{_spoken_name(repo['name'])} has no commits yet."
        lines = " ".join(
            f"{clean((c['commit']['message'].splitlines() or ['(no message)'])[0])}, "
            f"{_ago(c['commit']['author']['date'])}."
            for c in commits
        )
        return f"Latest on {_spoken_name(repo['name'])}: {lines}"

    def issues(self, spoken: str, pulls: bool = False) -> str:
        repo = self._repo(spoken)
        name = _spoken_name(repo["name"])
        if pulls:
            items = self._get(f"repos/{repo['full_name']}/pulls", state="open", per_page=LISTED)
            kind = "pull request"
        else:
            items = [i for i in self._get(f"repos/{repo['full_name']}/issues", state="open", per_page=20)
                     if "pull_request" not in i][:LISTED]
            kind = "issue"
        if not items:
            return f"{name} has no open {kind}s."
        titles = " ".join(f"{n}, {clean(i['title'])}." for n, i in enumerate(items, 1))
        count = f"{len(items)} open {kind}{'s' if len(items) != 1 else ''}"
        return f"{name} has {'at least ' if len(items) == LISTED else ''}{count}. {titles}"

    def stats(self, spoken: str) -> str:
        repo = self._repo(spoken)
        stars, forks = repo.get("stargazers_count", 0), repo.get("forks_count", 0)
        return (f"{_spoken_name(repo['name'])} has {stars} star{'s' if stars != 1 else ''} and "
                f"{forks} fork{'s' if forks != 1 else ''}. Last updated {_ago(repo.get('pushed_at', ''))}.")

    def readme(self, spoken: str) -> str:
        """README text of one of your repos, for the LLM to explain."""
        repo = self._repo(spoken)
        try:
            body = self._get(f"repos/{repo['full_name']}/readme")
        except GitHubError:
            description = repo.get("description") or "no description"
            return f"Repo {repo['name']} has no README. Its description: {description}"
        text =base64.b64decode(body.get("content", "")).decode("utf-8", "replace")
        return f"README of the {repo['name']} repo:\n\n{text[:MAX_README_CHARS]}"
