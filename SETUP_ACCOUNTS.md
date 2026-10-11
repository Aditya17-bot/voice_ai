# Connecting Gmail and GitHub (read-only)

Both are optional. Each takes a few minutes, once. The assistant can only
**read**: Google and GitHub themselves refuse any change, because the logins
you create here only have read permissions. The logins are stored encrypted
for your Windows user in `%APPDATA%\voice_assistant\` (never in the project
folder), and you can remove them at any time:

```powershell
.venv\Scripts\python -m assistant.setup status
.venv\Scripts\python -m assistant.setup forget gmail
.venv\Scripts\python -m assistant.setup forget github
```

## Gmail (about 10 minutes)

You make your own small "app" in Google Cloud (free), so no one else is
involved in your login.

1. Open <https://console.cloud.google.com/> and sign in with the Gmail account
   you want the assistant to read.
2. Create a project (top bar → project picker → **New project**), e.g.
   `voice-assistant`.
3. **APIs & Services → Library**, search **Gmail API**, click **Enable**.
4. **Google Auth Platform** (or "OAuth consent screen"):
   - **Branding**: app name `Voice assistant`, your email as support and
     developer contact.
   - **Audience**: User type **External**, then under **Test users** add your
     own Gmail address.
5. **Clients → Create client** → Application type **Desktop app** → Create →
   **Download JSON** (a file named like `client_secret_….json`).
6. Run, with the path to that file:

   ```powershell
   .venv\Scripts\python -m assistant.setup gmail "C:\Users\you\Downloads\client_secret_XXXX.json"
   ```

   Your browser opens. Pick your account. Google warns **"Google hasn't
   verified this app"**: that's expected for an app you made yourself; click
   **Continue**. Allow **"Read your email messages and settings"** (the
   read-only permission). The terminal then says
   `✓ Gmail connected for you@gmail.com, read-only.`

If Google ever grants more than read-only access, setup refuses to save it.

**Every 7 days?** While the Google app is in *Testing*, Google ends the login
after 7 days and the assistant will say "Gmail asked to sign in again"; just
run step 6 again. To avoid that you can try **Audience → Publish app**; for an
app only you use, Google shows the same "unverified" warning but keeps you
signed in.

Things to say: "How many emails did I get today?", "Do I have any new
emails?", "What are my latest emails?", "Find the email from Amazon", "Any
emails about the internship?", then "Read the second one" and "What else does
it say?".

## GitHub (about 3 minutes)

1. On GitHub: **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. Name it `voice assistant`, pick an expiration (e.g. 90 days), Resource owner:
   **you**, Repository access: **All repositories** (or only the ones you want).
3. **Repository permissions**, set each to **Read-only**:
   **Contents**, **Issues**, **Pull requests** (Metadata becomes read-only by
   itself). Leave everything else at **No access**.
4. **Generate token** and copy it (it starts with `github_pat_`).
5. Run, and paste the token when asked (it isn't shown as you paste):

   ```powershell
   .venv\Scripts\python -m assistant.setup github
   ```

Classic tokens (`ghp_…`) are refused, because they can't be limited to
reading.

Things to say: "List my repos", "What's new on my voice AI repo?", "Any open
issues on voice AI?", "Any pull requests on portfolio?", "How many stars does
voice AI have?", "What is my stonks repo about?".
