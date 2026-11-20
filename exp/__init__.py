"""Git-branch-driven experiment tracking for this repo (ported from clip-JEPA-fine-tune).

The modules here are additive: train_net.py only ever talks to
`exp.tracker.get_tracker()`, which returns a no-op tracker when CAT-Seg is launched
the original way (run.sh, eval.sh, python train_net.py). Tracking is only active
when a run is started through `run_experiment.py`.
"""
