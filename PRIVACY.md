# Privacy

This is a personal, offline voice assistant that runs entirely on its
owner's own computer. There is no server, no account system and no analytics.

**Gmail (optional).** If you connect Gmail, the assistant asks Google only
for the read-only Gmail permission (`gmail.readonly`). It reads message
counts, senders, subjects and, when you ask, the text of one email. That
text is processed on your computer by a local language model and is never
sent anywhere else, stored, or shared. The assistant cannot send, delete,
label or change email: Google refuses it because of the read-only
permission.

**GitHub (optional).** With a read-only token, it reads your repositories,
commits, issues, pull requests and READMEs, on your computer only.

**Logins** are stored encrypted for your Windows user on your computer
(Windows DPAPI) and are never sent anywhere except to Google or GitHub to
read your own data. You can remove them at any time with
`python -m assistant.setup forget gmail` (or `github`), or by removing the
app's access in your Google account.

**Speech** is recognized on your computer; audio is never recorded to disk
or sent anywhere.

Questions: open an issue at <https://github.com/Aditya17-bot/voice_ai>.
