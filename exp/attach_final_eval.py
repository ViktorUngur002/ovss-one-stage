"""Attach end-of-run numbers (the six OVSS benchmarks) to the training run of a commit.

The tracked final evaluation (`scripts/exp_final_eval.sh`) does this by itself for
each benchmark. Use the command line for numbers that come from elsewhere:

    python -m exp.attach_final_eval --result ade150=36.6 --result pc59=60.9
    python -m exp.attach_final_eval --commit a1b2c3d4 --result voc20=96.7

They land as summary fields final_mIoU/<name> and never affect which checkpoint was "best".
"""
import argparse
import sys

from . import gitinfo, paths, runmap


def attach(branch, commit, results, prefix="final_mIoU"):
    """Write {name: value} onto the training run of (branch, commit). Returns the run id."""
    entries = runmap.load().get(branch, {})
    match = entries.get(commit) or next(
        (v for k, v in entries.items() if k.startswith(commit) and ":" not in k), None)
    if match is None:
        raise LookupError(f"no training run recorded for branch {branch!r} commit {commit!r} "
                          f"in {paths.RUN_MAP_PATH}")
    import wandb
    api = wandb.Api()
    entity = paths.WANDB_ENTITY or api.default_entity
    run = api.run(f"{entity}/{paths.WANDB_PROJECT}/{match['run_id']}")
    for name, value in results.items():
        run.summary[f"{prefix}/{name}"] = value
    run.summary.update()
    return match["run_id"], run.name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commit", default=None,
                        help="commit hash (full or short); default is current HEAD")
    parser.add_argument("--branch", default=None, help="default is the current branch")
    parser.add_argument("--result", action="append", required=True, metavar="NAME=VALUE",
                        help="repeatable, e.g. --result ade150=36.6")
    args = parser.parse_args()

    info = gitinfo.collect()
    results = {}
    for item in args.result:
        if "=" not in item:
            sys.exit(f"--result expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        results[name.strip()] = float(value)
    try:
        run_id, name = attach(args.branch or info["branch"], args.commit or info["commit"], results)
    except LookupError as exc:
        sys.exit(str(exc))
    print(f"attached {results} to run {run_id} ({name})")


if __name__ == "__main__":
    main()
