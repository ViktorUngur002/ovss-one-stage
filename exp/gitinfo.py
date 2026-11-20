"""Resolve the current git state. Re-derived fresh at every launch."""
import re
import subprocess

from .paths import REPO_ROOT, SLUG_MAX_LEN


class GitError(RuntimeError):
    pass


def _git(*args):
    proc = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def slugify(text, max_len=None):
    """Commit subject -> filesystem-safe folder slug."""
    max_len = max_len or SLUG_MAX_LEN
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "no-subject"


def sanitize_branch(branch):
    """Branch names may contain '/', which we do not want in a folder name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", branch)


def collect():
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit = _git("rev-parse", "HEAD")
    subject = _git("log", "-1", "--pretty=%s")
    body = _git("log", "-1", "--pretty=%b").strip()
    dirty_files = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]
    try:
        remote = _git("remote", "get-url", "origin")
    except GitError:
        remote = ""
    return {
        "branch": branch,
        "branch_safe": sanitize_branch(branch),
        "detached": branch == "HEAD",
        "commit": commit,
        "commit_short": commit[:8],
        "subject": subject,
        "body": body,
        "slug": slugify(subject),
        "dirty": bool(dirty_files),
        "dirty_file_count": len(dirty_files),
        "dirty_files": "\n".join(dirty_files[:50]),
        "remote": remote,
    }
