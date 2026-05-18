---
description: Cut a release — bump version, draft CHANGELOG, tag, push
argument-hint: <new-version e.g. 1.3.0>
---

You are cutting release v$ARGUMENTS of accrue.

The PyPI publish workflow (`.github/workflows/publish.yml`) **enforces that the git tag matches `pyproject.toml`'s version**. Get this right or publish fails.

## 1. Sanity-check the request

- Confirm we are on `main` and clean: `git status`, `git fetch`, `git log origin/main..HEAD` should be empty.
- Read the current version: `grep '^version' pyproject.toml`.
- Compare to the requested $ARGUMENTS — must be a strict semver bump (major / minor / patch).
- If $ARGUMENTS is lower than or equal to the current version, **stop**.

## 2. Gather what's changed

```bash
git log v<previous-version>..HEAD --oneline
```

Group commits by `feat:` / `fix:` / `docs:` / `refactor:` / `test:`. Drop trivial ones (formatting, comment-only).

## 3. Draft the CHANGELOG entry

If `CHANGELOG.md` exists, prepend a section. Otherwise create one. Format:

```markdown
## [$ARGUMENTS] - <YYYY-MM-DD>

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

Only include sections that have entries.

## 4. Bump version

Edit `pyproject.toml`'s `version = "..."` to `$ARGUMENTS`. That is the only file the publish workflow checks — but if `accrue/__init__.py` exposes a `__version__`, bump it too.

## 5. Verify

```bash
ruff check .
ruff format --check .
pytest -x -q
python -m build
```

The build step catches packaging regressions before the tag goes out.

## 6. Commit, tag, push

```bash
git add pyproject.toml CHANGELOG.md accrue/__init__.py
git commit -m "feat: bump version to $ARGUMENTS"
git tag v$ARGUMENTS
git push origin main
git push origin v$ARGUMENTS
```

Then create the GitHub Release — that's what triggers `publish.yml`:

```bash
gh release create v$ARGUMENTS --title "v$ARGUMENTS" --notes-from-tag
```

## 7. Watch it ship

```bash
gh run watch
```

If publish fails: the most likely cause is tag/version mismatch. Don't force-push — fix the version on `main`, retag, redo the release.
