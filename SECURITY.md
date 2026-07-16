# Security policy

Do not open a public issue containing credentials, private prompts, model-access tokens, customer
data, chat exports, editor databases, or generated artifacts that may include sensitive text.

Report a suspected leak privately through the repository owner's GitHub profile. Include the file or
commit path, but redact the credential value. Treat any credential committed to Git as exposed even
if a later commit deletes the file; rotation and history cleanup are both required.

Before contributing, run a current-tree secret scan and inspect staged files. The repository CI also
scans the full reachable history with Gitleaks.

The verified procedure for removing a committed private archive is documented in
`docs/security/history_cleanup.md`. It prepares a disposable filtered mirror and never pushes by
itself.
