"""The (branch, commit) -> wandb run_id map. Local file, single source of truth.

Untracked by git on purpose: it records what *this machine* has run, not
anything about the code itself.
"""
import json
import os
import tempfile
from datetime import datetime, timezone

from .paths import RUN_MAP_PATH


def load():
    if not RUN_MAP_PATH.exists():
        return {}
    try:
        with open(RUN_MAP_PATH) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        # A corrupt map must never block a launch; keep it aside and start clean.
        try:
            os.replace(RUN_MAP_PATH, str(RUN_MAP_PATH) + ".corrupt")
        except OSError:
            pass
        return {}


def _write(data):
    fd, tmp = tempfile.mkstemp(dir=str(RUN_MAP_PATH.parent), prefix=".runmap-")
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, RUN_MAP_PATH)


def lookup(branch, commit):
    return load().get(branch, {}).get(commit)


def record(branch, commit, run_id, slug, extra=None):
    data = load()
    entry = {
        "run_id": run_id,
        "slug": slug,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)
    data.setdefault(branch, {})[commit] = entry
    _write(data)
    return entry
