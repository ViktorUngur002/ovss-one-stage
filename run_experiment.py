#!/usr/bin/env python
"""Launch CAT-Seg training or evaluation with experiment tracking bound to the git commit.

Use this instead of calling train_net.py directly (same arguments):

    python run_experiment.py --config-file configs/eva_vitl_336.yaml --num-gpus 1 [KEY VALUE ...]

It re-derives branch and commit fresh at every launch, binds the run to a wandb run
id that is permanent for that commit (branch = wandb group), sets cfg.OUTPUT_DIR to
MyRuns-OVSS/<branch>/<commit8>_<slug>/, and always passes --resume for training, so a
relaunch continues from detectron2's last periodic checkpoint in that folder.

With --eval-only it is a separate job (its own folder <commit8>_<slug>__eval/<dataset>/
and its own wandb run "[eval]"), and each benchmark's numbers are also attached to the
training run of the same commit.
"""
import logging
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp import gitinfo, paths, storage                      # noqa: E402
from exp.tracker import Tracker, set_tracker                 # noqa: E402


def _has_flag(argv, flag):
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _has_opt(argv, key):
    return key in argv


def _test_dataset(argv):
    """The DATASETS.TEST value passed on the command line, e.g. ade20k_150_test_sem_seg."""
    if "DATASETS.TEST" not in argv:
        return None
    value = argv[argv.index("DATASETS.TEST") + 1]
    match = re.search(r"[\"']?([A-Za-z0-9_]+)[\"']?", value)
    return match.group(1) if match else None


def _banner(info, run_dir, mode):
    width = 78
    print("=" * width)
    print(f" branch   : {info['branch']}   (wandb group)")
    print(f" commit   : {info['commit']}")
    print(f" subject  : {info['subject']}")
    if info["body"]:
        first = info["body"].splitlines()[0]
        print(f" notes    : {first[:60]}{'...' if len(info['body']) > 60 else ''}")
    print(f" run dir  : {run_dir}")
    print(f" tree     : {'DIRTY - not reproducible from git alone' if info['dirty'] else 'clean'}")
    print(f" mode     : {mode}")
    try:
        import open_clip
        print(f" open_clip: {open_clip.__file__}")
    except Exception as exc:
        print(f" open_clip: not importable ({exc})")
    print("=" * width)


def main():
    argv = list(sys.argv[1:])
    info = gitinfo.collect()
    if info["detached"]:
        print("[exp] WARNING: detached HEAD - the branch/group name will be 'HEAD'")

    is_eval = _has_flag(argv, "--eval-only")
    if is_eval:
        from exp.d2_hooks import short_name
        tag = os.environ.get("EXP_EVAL_TAG") or short_name(_test_dataset(argv) or "eval")
        tracker_dir = storage.run_dir(info["branch_safe"], info["commit_short"], info["slug"] + "__eval")
        output_dir = tracker_dir / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = f"eval-only ({tag})"
    else:
        tracker_dir = output_dir = storage.run_dir(info["branch_safe"], info["commit_short"], info["slug"])
        if not _has_flag(argv, "--resume"):
            argv.insert(0, "--resume")          # continue from last_checkpoint if there is one
        last = output_dir / "last_checkpoint"
        mode = f"train, resume from {last.read_text().strip()}" if last.exists() else "train, fresh start"

    # detectron2 options are KEY VALUE pairs at the end of the command line
    if not _has_opt(argv, "OUTPUT_DIR"):
        argv += ["OUTPUT_DIR", str(output_dir)]
    else:
        output_dir = Path(argv[argv.index("OUTPUT_DIR") + 1])

    _banner(info, output_dir, mode)
    tracker = Tracker(info, tracker_dir, argv, launch_script=os.environ.get("EXP_LAUNCH_SCRIPT"),
                      job_type="eval" if is_eval else "train")
    set_tracker(tracker)

    import train_net                                          # noqa: E402  (registers datasets)
    from detectron2.engine import default_argument_parser, launch

    args = default_argument_parser().parse_args(argv)
    assert args.num_gpus == 1, "tracking is written for a single GPU (one process)"
    print("Command Line Args:", args)

    status = "finished"
    try:
        launch(train_net.main, args.num_gpus, num_machines=args.num_machines,
               machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "crashed"
        raise
    finally:
        try:
            tracker.finish(status)
        except Exception as exc:
            logging.error("[exp] finish() failed: %s", exc)


if __name__ == "__main__":
    main()
