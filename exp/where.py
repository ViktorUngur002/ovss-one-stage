"""Print the training run folder of the current commit (used by the eval script).

    python -m exp.where            -> /mnt/win-data/MasterDisertatie/MyRuns-OVSS/<branch>/<commit8>_<slug>
    python -m exp.where --commit a1b2c3d4
"""
import argparse
import sys

from . import gitinfo, paths, runmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default=None, help="default is current HEAD (looked up in the run map)")
    args = parser.parse_args()
    info = gitinfo.collect()
    if args.commit is None:
        print(paths.MYRUNS_ROOT / info["branch_safe"] / f"{info['commit_short']}_{info['slug']}")
        return
    entries = runmap.load().get(info["branch"], {})
    match = next((v for k, v in entries.items() if k.startswith(args.commit) and ":" not in k), None)
    if match is None:
        sys.exit(f"no training run recorded for commit {args.commit!r} on branch {info['branch']!r}")
    print(match["run_dir"])


if __name__ == "__main__":
    main()
