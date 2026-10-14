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
from dataclasses import dataclass, replace
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
from . import clock
from .github import GitHubError, NotARepo
from .gmail import GmailError, NoList
from .youtube import NoVideos


@dataclass(frozen=True)
class Intent:
    action: str  # a key of _ACTIONS / _EXPLAIN_ACTIONS, or "open_any"
    arg: str = ""
    loose: bool = False  # exact names only; on a miss, hand over to the LLM
    site: str = ""  # search: which site
    profile: str = ""  # Chrome profile name as spoken ("robert"), "" for the usual one


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


# "chrome as robert", "youtube with my vit profile", "gmail in robert's account".
# A bare "as" takes at most two words ("as fast as possible" isn't a profile).
_PROFILE_NAME = r"(?!(?:to|my|your|a|an|the)\b)[\w.]+(?: [\w.]+)?"
_WITH_PROFILE = re.compile(
    rf"(?P<thing>.+?) (?:as (?:the |my )?(?P<p1>{_PROFILE_NAME}?)(?:'s)?(?: profile| account)?"
    rf"|(?:with|in|using|on|under) (?:the |my )?(?P<p2>{_PROFILE_NAME}?)(?:'s)? (?:chrome )?(?:profile|account))"
)


def _open(m: re.Match) -> Intent | None:
    target = _KIND_SUFFIX.sub("", _target(m["target"]))
    with_profile = _WITH_PROFILE.fullmatch(target)
    if with_profile:
        thing = _KIND_SUFFIX.sub("", _target(with_profile["thing"]))
        profile = with_profile["p1"] or with_profile["p2"]
        if resolve_app(thing) == "chrome":
            return Intent("open_app", thing, profile=profile)
        if resolve_site(thing):
            return Intent("open_website", thing, profile=profile)
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


_SEARCH_SITES = r"(?P<site>youtube|github|google)"
# Words around a query that aren't part of it: "search for some lofi music videos".
_QUERY_FILLER = re.compile(r"^(?:for |about )?(?:some |a few |a |an |the )?|(?: please| for me)$")
# A Google search with no site named must not be about this computer, or the
# words would leave it: "search for my notes", "search for the budget file".
_LOCAL_QUERY = re.compile(
    r"^(?:me|it|this|that|these|those|here)\b|\b(?:my|files?|folders?|director(?:y|ies)|project|notes?)\b"
)
# "google is a company", "google drive": a statement or a product, not a search.
_NOT_A_GOOGLE_SEARCH = re.compile(
    r"^(?:is|was|are|has|had|does|did|will|can|says?|said|chrome|drive|docs|sheets|slides|calendar"
    r"|earth|meet|photos|play|home|assistant|account)\b"
)


# A profile at the end of a search needs the word "profile" or "account":
# "search youtube for lofi in robert's profile". A bare "as robert" could be
# part of the query.
_SEARCH_PROFILE = re.compile(
    rf" (?:as|with|in|using|on|under) (?:the |my )?(?P<p>{_PROFILE_NAME}?)(?:'s)? (?:chrome )?(?:profile|account)$"
)


def _query_and_profile(raw: str) -> tuple[str, str]:
    """'lofi in robert's profile' -> ('lofi', 'robert'); leading 'some', 'a', 'the' dropped."""
    profile = ""
    with_profile = _SEARCH_PROFILE.search(raw)
    if with_profile:
        profile, raw = with_profile["p"], raw[: with_profile.start()]
    return _QUERY_FILLER.sub("", raw).strip(), profile


def _search(m: re.Match) -> Intent | None:
    """'search youtube for lofi', 'look up rust tutorials on github', 'google pasta recipes'."""
    query, profile = _query_and_profile(m["q"])
    named = m.groupdict().get("site")
    site = named or "google"
    if not query:
        return None
    if site == "google" and _NOT_A_GOOGLE_SEARCH.match(query):
        return None
    if named is None and _LOCAL_QUERY.search(query):
        return None
    return Intent("search", query, site=site, profile=profile)


# ----- YouTube: play by search, pick a result by number -----------------

_ORDINALS_SPOKEN = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9, "tenth": 10, "10th": 10, "top": 1, "next": "next",
}
_CARDINALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, **{str(n): n for n in range(1, 11)},
}
_ORD = "|".join(_ORDINALS_SPOKEN)
_CARD = "|".join(_CARDINALS)
_PICK_VERB = r"(?:click|play|open|watch|start|put on|select|choose|pick|show me|read)(?: me)?(?: on)?"
_PICK_NOUN = r"(?: one| video| result| link| song| track| e ?mail| message)"
# Hand gesture control, by any of its names.
_CAMERA = r"(?:camera|webcam|hand control|hand gestures?|gestures?|hand tracking)"
# Things you watch or listen to: "find an anime video", "play some lofi music".
_MEDIA = r"(?:videos?|songs?|music|trailers?|podcasts?|playlists?|mix|movie trailers?)"


def _pick(m: re.Match) -> Intent:
    """'play the second one', 'read the third email', 'click on video 3', 'the next one'.

    "play"/"video" picks a video and "read"/"email" an email; otherwise
    ("open the second one") whichever list was read out last.
    """
    spoken = m["n"] or m["n2"] or m["n3"]
    number = _ORDINALS_SPOKEN.get(spoken) or _CARDINALS[spoken]
    words = m.group(0)
    if re.search(r"\b(?:e ?mails?|messages?|mail)\b", words):
        action = "read_email"
    elif re.search(r"^(?:play|watch|put on)\b|\b(?:video|song|track)\b", words):
        action = "pick_video"
    else:
        action = "pick"  # "read the second one": whichever list is open
    # A bare "the second one" might answer something else: only act if a
    # list is on the go.
    return Intent(action, str(number), loose=m["n3"] is not None)


def _play(m: re.Match) -> Intent | None:
    """'play lofi on youtube', 'open on youtube clash royale'."""
    query, profile = _query_and_profile(m["q"] or m["q2"] or m["q3"])
    return Intent("play_video", query, profile=profile) if query else None


def _media(m: re.Match) -> Intent | None:
    """No site named, but clearly something to watch: 'find an anime video'."""
    query, profile = _query_and_profile(m["q"])
    if not query or _LOCAL_QUERY.search(query):
        return None  # "find the video files" is about this computer
    if re.search(rf"\b(?:{_ORD})\b", query):
        return None  # "play the second song": a pick, not a search
    if m["verb"] not in ("play", "put on", "watch") and len(query.split()) < 2:
        return None  # "open music" may be a folder; "open music videos" is not
    if m["verb"] in ("find", "show me", "get"):
        return Intent("search", query, site="youtube", profile=profile)
    return Intent("play_video", query, profile=profile)


_MAIL = r"(?:e ?mails?|messages|mails?)"
_MAILS = r"(?:e ?mails|messages|mails)"  # a list; "the latest email" is one to read
# One of your GitHub repos, said so: "my voice ai repo", "the portfolio repo",
# "voice_ai on github". A bare "what's new on youtube" is not about GitHub.
_REPO = (
    r"(?:(?:my |the )?(?P<repo>.+?) (?:repo|repository)(?: on github)?"
    r"|my (?P<repo2>.+?)(?: on github)?|(?:the )?(?P<repo3>.+?) on github)"
)


def _repo(m: re.Match) -> str:
    return m["repo"] or m["repo2"] or m["repo3"]


def _gh(action: str) -> Callable[[re.Match], Intent]:
    """GitHub intent; loose when only "my X" said it's a repo ("issues on my mind")."""
    return lambda m: Intent(action, _repo(m), loose=m["repo2"] is not None)


# ----- looking into a named folder ----------------------------------------


def _look_in(m: re.Match) -> Intent | None:
    """'what's in the downloads folder', 'tell me what you see in patents'."""
    name = _target(m["name"] or m["name2"])
    if _FILE_NAME.match(name) or _PRONOUN_TARGET.fullmatch(name):
        return None  # a file, or "what's in it": other rules handle those
    return Intent("look_in", name, loose=not (m["kind"] or m["kind2"]))


HELP_TEXT = (
    "I can talk with you about anything, and on this computer I can: "
    "open Chrome, the terminal, VS Code or Claude. Open Gmail, YouTube or GitHub. "
    "Search YouTube, GitHub or Google, and play a YouTube video, then play the second one or the next one. "
    "Move between your folders in adi, Desktop and Downloads, list them, and explain a file or folder. "
    "Once you connect them, read your Gmail, like how many emails today, or emails from someone, "
    "and your GitHub repos, issues and commits. "
    "Remember things you tell me, and tell you the time and date. I only read, I never change your files. "
    "Say camera on to control windows with your hand: thumbs up for a menu, hold your palm still then move it "
    "to switch windows, two fingers for desktops, one finger for tabs, thumbs down to close a window. "
    "Hold Right Control to talk, and say stop to quit."
)


def _check(m: re.Match) -> Intent | None:
    """'check my email', 'check youtube', 'check the patents folder'.

    Not apps: "check the terminal" or "check chrome" more often means "look
    at what's already open" than "open another one". "check the weather"
    stays loose and falls through to the LLM.
    """
    intent = _open(m)
    if intent is None or intent.action == "open_app":
        return None
    if intent.action in ("go_to_folder", "read_file"):
        # Exact names only: "check the notes file" with no such file is a
        # question for the LLM, not a "can't find it".
        return replace(intent, loose=True)
    return intent


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
        r"|list (?:this|the current) folder"
        r"|(?:(?:give|show) me )?(?:a |the )?list of (?:the |all )?(?:the )?files(?: here)?",
        lambda m: Intent("list_folder"),
    ),
    (
        r"(?:what(?:'s| is| are)|tell me what(?:'s| is| you see)?|what do you see|what can you see"
        r"|show me what(?:'s| is)) (?:in|inside) (?:the |my )?(?P<name>.+?)(?P<kind> folder| directory)?"
        r"|(?:list|show me)(?: all)? (?:the )?(?:files|folders|contents|stuff) (?:in|of|inside) "
        r"(?:the |my )?(?P<name2>.+?)(?P<kind2> folder| directory)?",
        _look_in,
    ),
    (
        r"(?:find|open|go to|show me|take me to|navigate to|switch to)(?: the| a| my)? "
        r"(?:folder|directory) (?:named|called) (?P<name>.+)",
        lambda m: Intent("go_to_folder", _target(m["name"])),
    ),
    (
        r"what can you do|what do you do|help|help me|what are you able to do|what can i (?:say|ask)(?: you)?"
        r"|what are your (?:features|commands|skills|abilities)|how do i use you",
        lambda m: Intent("help"),
    ),
    (
        r"what(?:'s| is) the time(?: now)?|what time is it(?: now)?|tell me the time",
        lambda m: Intent("time"),
    ),
    (
        r"(?:what(?:'s| is) the time|what time is it|what's the time|tell me the time|time)"
        r"(?: right)?(?: now)? in (?P<place>[a-z .']+?)(?: right now| now)?",
        lambda m: Intent("time_in", m["place"]),
    ),
    # Weather can't be known offline: show Google's answer for it.
    (
        r"(?:what(?:'s| is)|how(?:'s| is)|check|tell me|show me)(?: me)? the weather(?: like| forecast)?"
        r"(?: (?:in|at|for) (?P<place>.+?))?(?: today| tomorrow| now| this week)?"
        r"|(?:will|is) it (?:going to )?rain(?: today| tomorrow)?(?: in (?P<place2>.+?))?",
        lambda m: Intent("search", " ".join(["weather", m["place"] or m["place2"] or ""]).strip(),
                         site="google"),
    ),
    (
        r"what(?:'s| is) (?:the |today's )?date(?: today)?|what day is (?:it|today)(?: today)?"
        r"|what's today|what is today",
        lambda m: Intent("date"),
    ),
    (
        r"(?:clear|reset|forget) (?:the |this |our )?(?:conversation|chat|chat history|history)"
        r"|(?:start )?(?:a )?new (?:conversation|chat)|start over|fresh start",
        lambda m: Intent("clear_chat"),
    ),
    # Memory: only the user's own words are stored. "Remember to buy milk" is
    # a reminder, which this isn't, so it's left to the conversation.
    (
        r"remember (?:that |this )?(?P<fact>(?!(?:when|what|how|who|where|why|me|it|that|to)\b)\S+ .+)",
        lambda m: Intent("remember", m["fact"]),
    ),
    (
        r"what do you (?:remember|know) about me|what (?:do you|did i ask you to) remember"
        r"|what(?:'s| is) in your memory|what have you (?:remembered|saved)",
        lambda m: Intent("recall"),
    ),
    (
        r"forget (?:everything|all of it|it all|all that|everything you (?:know|remember)|all your memories)",
        lambda m: Intent("forget_all"),
    ),
    (
        r"forget (?:that |about |the fact that )?(?P<fact>(?!(?:it|that|this|everything|about it|about that)$).+)",
        lambda m: Intent("forget", m["fact"]),
    ),
    # Gmail, read-only.
    (
        rf"how many (?:new )?{_MAIL} (?:did i get|have i (?:got|gotten|received)|came in|i got|do i have)"
        rf" today|how many (?:new )?{_MAIL}(?: did i get| have i got)? today"
        rf"|(?:did i get|do i have|have i got) (?:any )?(?:new )?{_MAIL} today",
        lambda m: Intent("email_today"),
    ),
    (
        rf"how many (?:unread |new )?{_MAIL} (?:do i have|are there|are unread)(?: in my inbox)?"
        rf"|how many unread {_MAIL}|(?:do i have|have i got|are there|any) (?:any )?(?:new |unread ){_MAIL}",
        lambda m: Intent("email_unread"),
    ),
    (
        rf"(?:what are|read|show|tell me|list|give me|check)(?: me)? (?:my )?(?:the )?(?:latest|recent|newest|last"
        rf"|new)(?: few)? {_MAILS}|what(?:'s| is) (?:new )?in my inbox|(?:read|check) my inbox",
        lambda m: Intent("email_latest"),
    ),
    (
        rf"(?:read|show|list|what are)(?: me)? my unread {_MAIL}",
        lambda m: Intent("email_latest", "unread"),
    ),
    (
        rf"(?:find|show|read|search for|look for|get|check for|are there|is there|did i get|do i have|any)"
        rf"(?: me)?(?: an?| any| the)? {_MAIL} from (?P<who>.+?)(?: today| this week| recently| lately)?",
        lambda m: Intent("email_from", m["who"]),
    ),
    (
        rf"(?:find|show|search for|look for|get|are there|is there|any)(?: me)?(?: an?| any| the)? {_MAIL}"
        rf" (?:about|regarding|mentioning|on) (?P<topic>.+)",
        lambda m: Intent("email_about", m["topic"]),
    ),
    (
        rf"(?:read|open|show)(?: me)? (?:the |my )?(?:latest|newest|last|most recent) {_MAIL}",
        lambda m: Intent("read_email", "latest"),
    ),
    # GitHub, read-only.
    (
        r"(?:list|show|what are|tell me)(?: me)? (?:all )?my (?:github )?(?:repos|repositories)(?: on github)?"
        r"|how many (?:github )?(?:repos|repositories) do i have|my github repos",
        lambda m: Intent("gh_repos"),
    ),
    (
        rf"(?:what's new|what is new|what changed|any updates?|latest commits?|recent commits?|last commits?)"
        rf"(?: on| in| to| for| with) {_REPO}",
        _gh("gh_new"),
    ),
    (
        rf"(?:are there |any |what are the |show me the |list the |how many )?(?:open )?issues"
        rf"(?: are there)?(?: on| in| for) {_REPO}",
        _gh("gh_issues"),
    ),
    (
        rf"(?:are there |any |what are the |show me the |list the |how many )?(?:open )?(?:pull requests|prs|p rs)"
        rf"(?: are there)?(?: on| in| for) {_REPO}",
        _gh("gh_pulls"),
    ),
    (
        rf"how many (?:stars|forks)(?: does| do| on| for| has)? {_REPO}(?: have| has| got)?",
        _gh("gh_stats"),
    ),
    (
        r"(?:what is|what's|explain|describe|tell me about|summari[sz]e) my (?P<repo>.+?) (?:repo|repository)"
        r"(?: about)?(?: on github)?",
        lambda m: Intent("gh_readme", m["repo"]),
    ),
    # Voice console: the assistant's own controls. "ui" ones need the window.
    (
        r"(?:go |switch to |turn on |use |start )?hands[ -]?free(?: mode)?(?: on)?|always listen",
        lambda m: Intent("ui", "hands_free_on"),
    ),
    (
        r"(?:go |switch |switch to |back to |use )?(?:back to )?push[ -]to[ -]talk(?: mode)?"
        r"|(?:turn )?(?:off hands[ -]?free|hands[ -]?free off)",
        lambda m: Intent("ui", "hands_free_off"),
    ),
    (
        rf"(?:turn on|switch on|start|enable|use) (?:the |my )?{_CAMERA}|(?:the )?{_CAMERA} on",
        lambda m: Intent("ui", "camera_on"),
    ),
    (
        rf"(?:turn off|switch off|stop|disable|close) (?:the |my )?{_CAMERA}|(?:the )?{_CAMERA} off",
        lambda m: Intent("ui", "camera_off"),
    ),
    (r"(?:go |make it |switch to |enter )?full ?screen(?: mode)?", lambda m: Intent("ui", "fullscreen_on")),
    (r"(?:exit|leave|close|turn off) full ?screen(?: mode)?", lambda m: Intent("ui", "fullscreen_off")),
    (
        r"(?:change|switch|swap|next)(?: the| your)? (?:face|look|visuali[sz]er)"
        r"|(?:show|use|switch to) the (?:orb|circuit board|circuit)(?: face)?",
        lambda m: Intent("ui", "next_face"),
    ),
    (r"(?:hide|minimi[sz]e)(?: yourself| the window| your window)?", lambda m: Intent("ui", "hide")),
    (r"(?:show|open) (?:yourself|the window|your window)|come back", lambda m: Intent("ui", "show")),
    # YouTube: pick from the last search, or play by search.
    (
        rf"{_PICK_VERB} (?:the )?(?P<n>{_ORD}){_PICK_NOUN}?(?: on youtube)?"
        rf"|{_PICK_VERB} (?:the )?(?:video|result|number|song|track|e ?mail|message) (?:number )?(?P<n2>{_CARD})"
        rf"|(?:the )?(?P<n3>{_ORD}) (?:one|video|song|e ?mail|message)",
        _pick,
    ),
    (
        r"what are the (?:videos|results|options)|(?:read|tell|give) me the (?:titles|results|videos|list)"
        r"|what (?:videos|results) (?:are there|did you find|came up)|what did you find",
        lambda m: Intent("list_videos"),
    ),
    (
        r"(?:play|watch|put on|start)(?: me)? (?P<q>.+?) on youtube|open (?P<q2>.+?) on youtube"
        r"|(?:play|watch|open|start|put on)(?: on| in) youtube (?P<q3>.+)",
        _play,
    ),
    (
        rf"(?P<verb>find|show me|get|play|put on|watch|open)(?: me)? "
        rf"(?P<q>(?:.+? )?{_MEDIA})",
        _media,
    ),
    (
        # Trailing words are fine: "explain this project in one sentence".
        r"(?:explain|describe|summari[sz]e|tell me about|what(?:'s| is)) "
        r"(?:this|the current) (?:folder|project|directory|repo)\b.*",
        lambda m: Intent("explain_folder"),
    ),
    # Searches: the query is the user's own words, never the LLM's.
    (rf"(?:search|look up|find|look for)(?: on| in)? {_SEARCH_SITES}(?: for)? (?P<q>.+)", _search),
    (rf"(?:search(?: for)?|look up|find|look for) (?P<q>.+?) (?:on|in) {_SEARCH_SITES}", _search),
    (r"(?P<site>google) (?P<q>.+)", _search),
    (r"search (?:the web |online |the internet )?for (?P<q>.+)", _search),
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
    (r"(?P<verb>check)(?: out| on)? (?P<target>.+)", _check),
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


# The whole utterance, alone: "stop", "goodbye". "Stop the music" is not a quit.
# Closers like "that's all" answer "anything else?", and Whisper turns noise
# into a bare "Bye.", so neither quits.
_QUIT = re.compile(r"(?:stop|quit|exit|goodbye|good bye|stop listening)(?: (?:jarvis|assistant))?")


def wants_to_quit(text: str) -> bool:
    """True if the user said only a quit phrase."""
    return bool(_QUIT.fullmatch(_prepare(text)))


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
    "open_app": lambda ws, i: ws.open_app(i.arg, i.profile),
    "open_here": lambda ws, i: ws.open_here(i.arg),
    "open_website": lambda ws, i: ws.open_website(i.arg, i.profile),
    "search": lambda ws, i: ws.search(i.site, i.arg, i.profile),
    "play_video": lambda ws, i: _listing(ws, "youtube", ws.youtube.play(i.arg, i.profile)),
    "pick_video": lambda ws, i: _listing(ws, "youtube", ws.youtube.pick(_number(i.arg))),
    "list_videos": lambda ws, i: _listing(ws, "youtube", ws.youtube.titles()),
    "email_today": lambda ws, i: ws.gmail.count_today(),
    "email_unread": lambda ws, i: ws.gmail.count_unread(),
    "email_latest": lambda ws, i: _listing(ws, "email", ws.gmail.latest(unread_only=i.arg == "unread")),
    "email_from": lambda ws, i: _listing(ws, "email", ws.gmail.from_sender(i.arg)),
    "email_about": lambda ws, i: _listing(ws, "email", ws.gmail.about(i.arg)),
    "gh_repos": lambda ws, i: ws.github.repos(),
    "gh_new": lambda ws, i: ws.github.whats_new(i.arg),
    "gh_issues": lambda ws, i: ws.github.issues(i.arg),
    "gh_pulls": lambda ws, i: ws.github.issues(i.arg, pulls=True),
    "gh_stats": lambda ws, i: ws.github.stats(i.arg),
    "help": lambda ws, i: HELP_TEXT,
    "time": lambda ws, i: clock.now_local(),
    "date": lambda ws, i: clock.today(),
    "time_in": lambda ws, i: clock.time_in(i.arg) or f"I don't know the time zone for {i.arg}.",
}
# Actions whose result is material for the LLM to explain.
_EXPLAIN_ACTIONS: dict[str, Callable[[Workspace, Intent], str]] = {
    "explain_folder": lambda ws, i: ws.explain_folder(),
    "read_file": lambda ws, i: ws.read_file(i.arg, exact=i.loose),
    "read_email": lambda ws, i: _listing(
        ws, "email", ws.gmail.read_latest() if i.arg == "latest" else ws.gmail.read(_number(i.arg))
    ),
    "gh_readme": lambda ws, i: ws.github.readme(i.arg),
}
# Actions that read out a numbered list, or pick from one: "the second one"
# keeps meaning that list until something else happens.
LIST_ACTIONS = {"play_video", "pick_video", "list_videos", "pick", "read_email", "email_latest",
                "email_from", "email_about"}


def _number(arg: str) -> int | str:
    return arg if arg == "next" else int(arg)


def _listing(ws: Workspace, kind: str, said: str) -> str:
    """Remember which list was read out last, for a bare "the second one"."""
    ws.last_list = kind
    return said


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
    except NoVideos as exc:
        if intent.loose:
            return None  # a bare "the second one" with no search going
        raise ToolError(str(exc)) from None
    except (GmailError, GitHubError) as exc:
        if intent.loose and (isinstance(exc, (NoList, NotARepo)) or intent.action.startswith("gh_")):
            return None  # a bare "the second one", or "issues on my mind": not ours
        raise ToolError(str(exc)) from None
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
    if intent.action == "pick":
        # "open the second one": from whichever list was read out last.
        if workspace.last_list == "email":
            return _run(replace(intent, action="read_email"), workspace)
        return _run(replace(intent, action="pick_video"), workspace)
    if intent.action == "look_in":
        if intent.loose:
            # "what's in downloads" or "what's in the readme": exact names only.
            try:
                workspace.go_to_folder(intent.arg, exact=True)
            except NotFound:
                return Outcome(workspace.read_file(intent.arg, exact=True), explain=True)
        else:
            workspace.go_to_folder(intent.arg)
        return Outcome(workspace.list_folder(), explain=False)
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
                "VS Code, Claude, Gmail, YouTube, GitHub, or one of your folders."
            ) from None
    raise ValueError(f"unknown action {intent.action!r}")


# ----- follow-ups ---------------------------------------------------------

_ORDINALS = {"first": 0, "1st": 0, "former": 0, "second": 1, "2nd": 1, "latter": 1, "other": 1, "last": 1}
_YES = {"yes", "yeah", "yep", "sure", "ok", "okay"}
_NO = {"no", "not", "nope", "nah", "don't", "doesn't", "wrong", "neither", "never"}
# Words an answer may carry around the choice itself: "yes, the one in work".
_ANSWER_FILLER = _YES | {"the", "one", "in", "i", "meant", "mean", "that", "please"}
MAX_ANSWER_WORDS = 6


def is_yes(text: str) -> bool:
    """A short, plain yes: "yes", "yeah, do it". Anything with a no in it isn't."""
    words = _prepare(text).split()
    return (
        0 < len(words) <= MAX_ANSWER_WORDS
        and not _NO & set(words)
        and bool(_YES & set(words))
        and set(words) <= _ANSWER_FILLER | {"do", "it", "go", "ahead", "confirm", "sure"}
    )


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
