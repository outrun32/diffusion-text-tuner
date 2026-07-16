# Removing the historical Copilot handoff archive

The reachable Git history contains a deleted 25.7 MB handoff bundle with Copilot chat/session and
VS Code state. A case-sensitive credential scan did not find AWS, Hugging Face, OpenAI, or GitHub
token shapes, so the issue is privacy/IP exposure rather than a confirmed credential leak.

Deleting the file in a later commit is insufficient. Every branch and tag that contains the blob
must be rewritten, then GitHub must receive force-updated heads and tags.

## Preparation

1. Commit or back up the current working tree separately. The cleanup command reads committed Git
   history only; it does not include uncommitted files.
2. Announce a maintenance window. Existing clones must stop pushing until the rewrite finishes.
3. Temporarily allow force pushes on protected branches, if applicable.
4. Record open pull requests, release tags, GitHub Pages source branch, and any external commit links.

## Build a filtered mirror without pushing

Run this outside the repository directory. Set `REPO` to the checkout first; the script itself uses
only the standard library, so it does not need the repository as the current working directory.

```bash
REPO=/absolute/path/to/diffusion-text-tuner
cd /tmp
python "$REPO/scripts/prepare_history_cleanup.py" \
  --source https://github.com/outrun32/diffusion-text-tuner.git \
  --work-dir /tmp/diffusion-text-tuner-filtered.git
```

The command:

- creates a bare mirror clone;
- writes a full pre-filter Git bundle and SHA-256 beside it;
- removes `outputs/handoff/` and `scripts/export_handoff_bundle.sh` with pinned
  `git-filter-repo==2.47.0 --sensitive-data-removal`;
- confirms that all branch/tag names remain present;
- runs `git fsck` and verifies that forbidden paths are unreachable;
- emits, but never executes, separate force-push commands for heads and tags.

If a clone URL contains HTTP credentials or a signed query, the script uses it only for the clone,
then removes the credentials from the mirror remote, JSON report, console diagnostics, and printed
push commands. The printed commands rely on the operator's Git credential helper or SSH setup.

Keep the bundle offline until the rewrite is accepted. It still contains the private history.

## Review before push

```bash
git -C /tmp/diffusion-text-tuner-filtered.git show-ref --heads --tags
git -C /tmp/diffusion-text-tuner-filtered.git fsck --full --no-dangling
git -C /tmp/diffusion-text-tuner-filtered.git rev-list --objects --branches --tags \
  | rg 'outputs/handoff|export_handoff_bundle' && exit 1 || true
```

Compare the cleanup report's ref names with GitHub. Commit SHA changes are expected; missing branch
or tag names are not.

## Push only after explicit approval

Run the two commands printed in `push_commands`. Do not use `git push --mirror`: it can delete remote
service refs that were not cloned as ordinary heads/tags.

After pushing:

1. re-enable branch protection;
2. rerun `make history-audit` against a fresh clone;
3. rerun Gitleaks and CI;
4. verify the GitHub Pages branch/deployment and release tags;
5. ask collaborators to archive old clones and re-clone. Rebasing old branches can reintroduce the
   removed blob;
6. contact GitHub Support if a cached raw-object URL remains accessible after refs and garbage
   collection have updated.

The offline backup bundle must never be uploaded or committed.
