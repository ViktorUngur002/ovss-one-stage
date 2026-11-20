"""Run folder layout on the big local disk, plus the weights-only cloud push.

    MyRuns-OVSS/<branch>/<commit8>_<slug>/        (= detectron2's cfg.OUTPUT_DIR)
        model_XXXXXXX.pth   last periodic checkpoint (crash recovery, newest only)
        last_checkpoint     detectron2's pointer used by --resume
        model_best.pth      best by the in-training eval (detectron2 BestCheckpointer)
        model_final.pth     end of training
        best_weights.pt     weights only, fp16 - this is what goes to the cloud
        best_metrics.json   exp_state.json (resume bookkeeping)
        config.yaml, log.txt, metrics.json   written by detectron2 itself
        config/             launch script and argv, for provenance
        final_eval/<dataset>/   the six OVSS benchmarks after training
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import paths

STATE_FILENAME = "exp_state.json"


def run_dir(branch_safe, commit_short, slug, create=True):
    path = paths.MYRUNS_ROOT / branch_safe / f"{commit_short}_{slug}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def remote_dir(branch_safe, folder_name):
    return f"{paths.RCLONE_REMOTE.rstrip('/')}/{branch_safe}/{folder_name}"


def _atomic_json(path, data):
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), prefix=".tmp-")
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def read_state(directory):
    path = Path(directory) / STATE_FILENAME
    if not path.exists():
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def write_state(directory, state):
    _atomic_json(Path(directory) / STATE_FILENAME, state)


def write_json(path, data):
    _atomic_json(path, data)


def copy_into(directory, src, dest_name=None):
    """Copy a provenance file (launch script, config) into the run folder."""
    src = Path(src)
    if not src.exists():
        return None
    dest = Path(directory) / "config" / (dest_name or src.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def link_or_copy(src, dest):
    """Hardlink when possible - a 5 GB checkpoint then costs no extra space.

    Both paths live on the same filesystem here, and torch.save never mutates
    an existing file in place, so the link can never be changed underneath us.
    """
    src, dest = Path(src), Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        os.link(src, tmp)
        mode = "hardlink"
    except OSError:
        shutil.copy2(src, tmp)
        mode = "copy"
    os.replace(tmp, dest)
    return mode


def export_weights(state_dict, dest, dtype=None, meta=None):
    """Write a weights-only checkpoint: no optimizer state, optionally fp16.

    Small enough to be worth pushing to a 15 GB cloud account; the resumable
    detectron2 files (with optimizer state) stay local.
    """
    dtype = (dtype or paths.CLOUD_WEIGHTS_DTYPE).lower()
    cast = torch.float16 if dtype == "fp16" else torch.float32
    out = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            value = value.detach().to("cpu")
            if value.is_floating_point():
                value = value.to(cast)
        out[key] = value
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save({"state_dict": out, "meta": meta or {}}, tmp)
    os.replace(tmp, dest)
    return dest, dest.stat().st_size


def rclone_available():
    return shutil.which("rclone") is not None


def rclone_push_async(local_dir, remote_path, includes):
    """Start an rclone copy in the background and return the Popen handle.

    Never blocking: an 860 MB upload on a home connection takes minutes and
    must not stall the training loop at an evaluation.
    """
    if not paths.RCLONE_ENABLED:
        return None
    if not rclone_available():
        logging.warning("[exp] rclone not installed - skipping cloud push")
        return None
    cmd = ["rclone", "copy", str(local_dir), remote_path, *paths.RCLONE_FLAGS]
    for pattern in includes:
        cmd += ["--include", pattern]
    logging.info("[exp] cloud push: %s", " ".join(cmd))
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        logging.warning("[exp] could not start rclone: %s", exc)
        return None


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
