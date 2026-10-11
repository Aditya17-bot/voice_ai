"""One-time account setup for read-only Gmail and GitHub.

    python -m assistant.setup gmail path\\to\\client_secret.json
    python -m assistant.setup github
    python -m assistant.setup status
    python -m assistant.setup forget gmail|github

Gmail: signs in through Google in your browser and asks only for the
read-only Gmail permission; a login with any wider permission is refused.
For the moment of sign-in, a one-request listener on 127.0.0.1 receives
Google's answer, then closes. GitHub: takes a fine-grained token created with
read-only permissions (classic tokens, which can't be limited to reading,
are refused). Both are stored encrypted for your Windows user only.
"""

import base64
import getpass
import hashlib
import http.server
import json
import secrets
import sys
import time
import urllib.parse
import webbrowser
from pathlib import Path

import requests

from . import credentials, github, gmail

# Windows consoles default to cp1252, which can't print ✓ / ✗.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
SIGN_IN_TIMEOUT_S = 300


def _fail(message: str) -> None:
    print(f"\n✗ {message}")
    sys.exit(1)


def _sign_in(state: str, open_browser) -> tuple[str, str]:
    """Listen on 127.0.0.1 for Google's redirect; return (code, redirect_uri).

    `open_browser(redirect_uri)` is called once the listener has its port.
    """
    result: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        timeout = 5  # a connection that sends nothing can't stall the sign-in

        def do_GET(self) -> None:  # noqa: N802 — http.server's name
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
            if query.get("state") == state:
                result.update(query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            done = "code" in result
            message = "Signed in. You can close this tab." if done else "Sign-in didn't finish. Try again."
            self.wfile.write(f"<p style='font:18px sans-serif'>{message}</p>".encode())

        def log_message(self, *args) -> None:
            pass  # keep the codes out of the console

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 1
    redirect_uri = f"http://127.0.0.1:{server.server_port}"
    open_browser(redirect_uri)
    deadline = time.time() + SIGN_IN_TIMEOUT_S
    try:
        while "code" not in result and "error" not in result and time.time() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if "error" in result:
        _fail(f"Google said: {result['error']}")
    if "code" not in result:
        _fail("Timed out waiting for the sign-in.")
    return result["code"], redirect_uri


def setup_gmail(client_file: str) -> None:
    try:
        client = json.loads(Path(client_file).read_text(encoding="utf-8"))["installed"]
        client_id, client_secret = client["client_id"], client["client_secret"]
    except (OSError, ValueError, KeyError):
        _fail("That isn't a Desktop-app OAuth client file (client_secret_….json, with an 'installed' section).")
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)

    def open_browser(redirect_uri: str) -> None:
        url = AUTH_URL + "?" + urllib.parse.urlencode({
            "client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code",
            "scope": gmail.SCOPE, "access_type": "offline", "prompt": "consent",
            "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
        })
        print("Opening Google sign-in in your browser. If it doesn't open, paste this link:\n")
        print(url, "\n")
        webbrowser.open(url)

    code, redirect_uri = _sign_in(state, open_browser)
    response = requests.post(gmail.TOKEN_URL, data={
        "code": code, "client_id": client_id, "client_secret": client_secret,
        "redirect_uri": redirect_uri, "grant_type": "authorization_code", "code_verifier": verifier,
    }, timeout=15)
    if not response.ok:
        _fail(f"Google refused the sign-in ({response.status_code}).")
    tokens = response.json()
    granted = set(tokens.get("scope", "").split())
    if granted != {gmail.SCOPE}:
        # Hand the broader login straight back to Google so it can't be used.
        requests.post(REVOKE_URL, params={"token": tokens.get("refresh_token") or tokens.get("access_token")},
                      timeout=15)
        _fail(f"Google granted {sorted(granted)}, not only read-only Gmail. Not saved, and revoked.")
    if "refresh_token" not in tokens:
        _fail("Google didn't send a long-term login. Remove the app's access in your Google account and retry.")
    profile = requests.get(f"{gmail.API}/profile", timeout=15,
                           headers={"Authorization": f"Bearer {tokens['access_token']}"})
    if not profile.ok:
        _fail("Signed in, but Gmail didn't answer. Is the Gmail API enabled in your Google Cloud project?")
    credentials.save(gmail.SECRET_NAME, {
        "client_id": client_id, "client_secret": client_secret, "refresh_token": tokens["refresh_token"],
    })
    print(f"✓ Gmail connected for {profile.json().get('emailAddress')}, read-only.")


def setup_github() -> None:
    print("Paste a fine-grained GitHub token with read-only permissions (the input is hidden).")
    token = getpass.getpass("Token: ").strip()
    if token.startswith(("ghp_", "gho_", "ghu_")):
        _fail("That's a classic token, which can't be limited to reading. Make a fine-grained one (github_pat_…).")
    if not token.startswith("github_pat_"):
        _fail("That doesn't look like a fine-grained GitHub token (they start with github_pat_).")
    response = requests.get(f"{github.API}/user", timeout=15, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    if not response.ok:
        _fail(f"GitHub didn't accept the token ({response.status_code}).")
    # GitHub doesn't tell an app what a fine-grained token may do, so ask.
    print("\nThis app can't check the token's permissions itself. When you made it, were")
    print("Contents, Issues and Pull requests set to Read-only, and everything else to No access?")
    if input("Type 'read only' to confirm: ").strip().lower() != "read only":
        _fail("Not saved. Make a token with only read-only permissions, then run this again.")
    credentials.save(github.SECRET_NAME, {"token": token})
    print(f"✓ GitHub connected for {response.json().get('login')}.")


def status() -> None:
    for name in (gmail.SECRET_NAME, github.SECRET_NAME):
        print(f"{name}: {'saved' if credentials.load(name) else 'not set up'}")
    print("(Saved logins can still have expired: a Gmail app left in Testing signs out after 7 days.)")


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "gmail":
        setup_gmail(argv[1])
    elif argv[:1] == ["github"]:
        setup_github()
    elif argv[:1] == ["status"]:
        status()
    elif len(argv) == 2 and argv[0] == "forget" and argv[1] in (gmail.SECRET_NAME, github.SECRET_NAME):
        print("Removed." if credentials.delete(argv[1]) else "Nothing was stored.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
