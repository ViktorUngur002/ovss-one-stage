"""Machine-specific configuration, all overridable through the environment.

Defaults match this machine (mojkomp). Override them in `exp/env.sh`, which is
untracked - see `exp/env.sh.example`.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _flag(name, default):
    return os.environ.get(name, str(default)).strip().lower() not in ("0", "false", "no", "")


# Where every run folder lives. Big local disk, never the root partition.
# Kept apart from clip-JEPA-fine-tune's MyRuns: CAT-Seg runs are a different stage.
MYRUNS_ROOT = Path(os.environ.get("EXP_MYRUNS_ROOT", "/mnt/win-data/MasterDisertatie/MyRuns-OVSS"))

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "OVSS One stage arch experiments")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None

# The governing metric for "best checkpoint": the mean of these keys out of the
# flattened results of detectron2's in-training evaluation (cfg.DATASETS.TEST,
# COCO-Stuff val in the upstream config). The six OVSS benchmarks are evaluated
# after training and never pick the checkpoint.
BEST_METRIC_KEYS = tuple(
    k.strip() for k in os.environ.get("EXP_BEST_METRIC_KEYS", "sem_seg/mIoU").split(",") if k.strip()
)
BEST_METRIC_NAME = os.environ.get("EXP_BEST_METRIC_NAME", "intrain_mIoU")

# Cloud copy: a weights-only export of the best checkpoint, not the resumable files.
RCLONE_ENABLED = _flag("EXP_RCLONE_ENABLED", 1)
RCLONE_REMOTE = os.environ.get("EXP_RCLONE_REMOTE", "gdrive:MyRuns-OVSS")
RCLONE_FLAGS = os.environ.get(
    "EXP_RCLONE_FLAGS",
    "--drive-chunk-size 128M --transfers 2 --retries 3 --drive-stop-on-upload-limit",
).split()
CLOUD_WEIGHTS_ENABLED = _flag("EXP_CLOUD_WEIGHTS_ENABLED", 1)
CLOUD_WEIGHTS_DTYPE = os.environ.get("EXP_CLOUD_WEIGHTS_DTYPE", "fp16")   # fp16 | fp32

# detectron2 writes model_XXXXXXX.pth every SOLVER.CHECKPOINT_PERIOD. Keep only the
# newest N of them (crash recovery); model_final.pth and model_best.pth are always kept.
KEEP_PERIODIC_CHECKPOINTS = int(os.environ.get("EXP_KEEP_PERIODIC_CHECKPOINTS", "1"))

# How often (iterations) training scalars go to wandb; detectron2's own console/json
# writers keep their default period of 20.
LOG_EVERY_N_ITERS = int(os.environ.get("EXP_LOG_EVERY_N_ITERS", "20"))

RUN_MAP_PATH = REPO_ROOT / ".wandb_run_map.json"
SLUG_MAX_LEN = int(os.environ.get("EXP_SLUG_MAX_LEN", "60"))
