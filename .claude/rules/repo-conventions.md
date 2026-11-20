# Repo conventions for code we hand to the user

Extend what exists; do not build parallel infrastructure.

- **New behaviour = a new config key** in `cat_seg/config.py` (`add_cat_seg_config`), read where it is
  used, with a default that reproduces the upstream CAT-Seg/DeCLIP behaviour exactly. The authors'
  configs (`configs/*.yaml`) must keep working, since the reproduction depends on them.
- **Model-side changes** (dense mode, block-8 exit / placements, feature taps) go in CAT-Seg's EVA-CLIP
  copy `cat_seg/src/open_clip/eva_clip/eva_vit_model.py` next to `encode_dense` / `ss_attn`, following
  the existing `mode`-string dispatch, and are selected from `cat_seg_model.py` through a config key —
  never by editing the hard-coded `mode='csa'` silently. Stage-1 code (clip-JEPA-fine-tune) is the
  reference for these; port it, do not import it.
- **Losses / training logic** stay in `cat_seg/cat_seg_model.py` and the head; anything returned in
  the model's loss dict is summed by detectron2 and logged, so never put diagnostics there.
- **Optimizer**: param groups are built in `train_net.py` `Trainer.build_optimizer` (`backbone` and
  `clip_model` multipliers). A new trainable module needs its own group/multiplier there.
- **Frozen parts**: `CATSeg.frozen_backbone` decides what in CLIP trains; change it only through a
  config key.
- **Tracking** goes through `exp.tracker.get_tracker()` (no-op unless `run_experiment.py` launched the
  run); detectron2 glue lives in `exp/d2_hooks.py`. Never import wandb in model code.
- **New launch configs**: copy `scripts/exp_catseg_eva_vitl336.sh` to a new `scripts/exp_*.sh`, keep the
  explanatory comment block (what the run is, what differs from the last one), pass config changes as
  `KEY VALUE` pairs or a new `configs/*.yaml`; never set `OUTPUT_DIR` (the wrapper does).
- **Datasets**: root from `DETECTRON2_DATASETS` (exp/env.sh); never hard-code new dataset paths.
- Style: plain PyTorch and detectron2 idioms, sparse comments that explain *why*, no drive-by
  refactors of upstream code, no type-annotation/docstring overhauls.
- Before proposing code, state its VRAM and step-time cost at batch 1 (384 crops, CLIP at 336).
