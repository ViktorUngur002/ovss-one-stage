# CLAUDE.md

Master's thesis codebase, **stage 2 (OVSS)**: CAT-Seg trained on top of a DeCLIP-style backbone for
open-vocabulary semantic segmentation. Baseline: the authors' official `DeCLIP_CATSeg` branch
(github.com/xiaomoguhz/DeCLIP, commit `bb2990c`), copied here as one commit with the single-GPU
fixes needed on this machine. Stage 1 (fine-tuning the backbone, DeCLIP + JEPA, "Placement E" runs
TN/TW) lives in the sibling repo `clip-JEPA-fine-tune`.

**Current goal:** reproduce the authors' CAT-Seg + DeCLIP EVA-L/14-336 numbers with their released
backbone (README: ADE847 17.96, PC459 25.89, ADE150 40.59, PC59 63.79, VOC20 97.76, VOC21 83.67),
on one GPU. Only after that do our own backbones (TN, TW and their test-time placements) get plugged in.

Work branch: `Initial_run_DeCLIP_CATSeg_results_reproduction`. Work on the branch the user names;
never switch branches on your own.

## How we work together (hard rules)

1. **Do not write or edit code unless the user explicitly asks you to in that message.** By default,
   hand code back in the chat and wait for the user to type it in. Approval for one edit does not
   carry over to the next. (Docs/config for Claude itself, like this file, are the exception only
   when the user asks for them.)
2. **Every code snippet comes with:**
   - the exact location: file path, the function/class it goes in, and the anchor line it goes
     before/after (quote the existing line so it can be found with search), or "replace lines X–Y"
     with the old text shown;
   - a plain-language explanation of what it does and why, short enough to read in one pass.
     No unexplained jargon; say what each tensor is and its shape.
3. **One chunk at a time.** Never bundle the next change into the current one.
4. **Hyper-parameters are open until decided.** LR, iterations, batch, crop size, eval period,
   CLIP multiplier, etc.: before fixing one, check what CAT-Seg / DeCLIP actually use, adapt it to
   *our* setup (single 20 GB GPU, batch 1), and give a reasoned recommendation. Track decisions in
   `.claude/rules/ovss-decisions.md`.
5. **Flag risks and better alternatives proactively**, but every flag must be argued: what breaks,
   why (mechanism, numbers, citation or code line), and what you would do instead. No generic warnings.
6. **Review loop after the user writes code:** when asked to check, read the actual files, verify
   shapes, gradient flow, dtype/autocast, device, frozen/trainable params, and that the change
   matches what was agreed. Report concrete problems with `file:line`.
7. **Never commit or push — the user always does commits themselves** (unless they explicitly ask
   in that message). Do not launch training either. Each commit becomes a wandb run (see below), so
   commit timing is theirs; when a chunk is done, just say it is ready to commit.

## WandB monitoring protocol

After a piece is implemented and checked, offer **monitoring options** for that piece: a short list,
each option explained simply (what you'd see in wandb, what "healthy" vs "broken" looks like, and the
cost in time/VRAM/log size). The user picks; only then provide (or, if asked, write) the logging
code. Technical details: `.claude/rules/wandb-monitoring.md`; tracking package: `exp/README.md`.

## Compute budget

- One AMD GPU (ROCm, `HIP_VISIBLE_DEVICES`), **20 GB VRAM**, conda env **`DeCLIP_CATSeg`**
  (torch 2.10 + ROCm 7.0, detectron2 0.6, wandb). No xformers (`xattn` is off in the EVA configs).
- The paper trains with 4 images per batch on 4 GPUs; here `SOLVER.IMS_PER_BATCH 1` x `SOLVER.GRAD_ACCUM_STEPS 4`
  under fp16 AMP = the paper's batch 4, `BASE_LR` 2e-4, MAX_ITER 80000 (decided 2026-11-22; batch 2+ does not
  fit even with AMP). See `.claude/rules/ovss-decisions.md`.
- Training/launching happens on the Linux box (`/home/viktor/...`, `/mnt/win-data/...`). This Mac
  is for editing; do not try to run training here.
- Every in-training eval (COCO-Stuff val, 5000 images) and every final benchmark costs wall time;
  cost anything added in VRAM and step time before proposing it.

## Repo map

| Area | Where |
|---|---|
| Training / eval entry | `train_net.py` (detectron2 `DefaultTrainer` subclass, optimizer param groups, evaluators) |
| Tracked launch | `run_experiment.py` + `scripts/exp_catseg_eva_vitl336.sh` (train), `scripts/exp_final_eval.sh` (6 benchmarks) |
| Upstream launch (untracked) | `run.sh`, `eval.sh` — unchanged |
| Model | `cat_seg/cat_seg_model.py` (`CATSeg`: CLIP dense features → head; `frozen_backbone`) |
| Head / predictor | `cat_seg/modeling/heads/cat_seg_head.py`, `cat_seg/modeling/transformer/cat_seg_predictor.py` (builds EVA-CLIP from `CACHE_DIR`, text features) |
| Backbone copy | `cat_seg/src/open_clip/eva_clip/eva_vit_model.py` (`encode_dense`, `ss_attn`) — CAT-Seg's own EVA-CLIP copy, not stage 1's |
| Config | `configs/config.yaml` (base), `configs/eva_vitl_336.yaml`, `cat_seg/config.py` (`add_cat_seg_config`) |
| Datasets | `cat_seg/data/datasets/register_*.py`, root `DETECTRON2_DATASETS` (exp/env.sh), class lists `datasets/*.json` |
| Tracking | `exp/` (`tracker.py`, `d2_hooks.py`), see `exp/README.md` |

Experiment bookkeeping: branch = wandb group, commit = one wandb run (subject = run name, body =
notes). Runs land in `/mnt/win-data/MasterDisertatie/MyRuns-OVSS/<branch>/<commit8>_<slug>/`.

## Facts about the code (verified; re-check if the files changed)

- Backbone: EVA02-CLIP-L-14-336, the authors' `DeCLIP_EVA-L_DINOv2-L_csa_0.05_2.0.pt` from
  `MODEL.SEM_SEG_HEAD.CACHE_DIR` (`/home/viktor/MasterDisertatie/checkpoints/`). CLIP input is
  resized to 336 → 24 × 24 patches (`FEATURE_RESOLUTION [24, 24]`); training crops are 384.
- `cat_seg_model.py` hard-codes `encode_dense(..., mode='csa')` twice (training forward and
  sliding-window inference). Our stage-1 checkpoints are qq-mode; the README says to switch both.
- For EVA, `frozen_backbone` ignores `CLIP_FINETUNE`: it trains `q_proj`/`v_proj` of all visual
  blocks and `in_proj_weight` of the text transformer, everything else in CLIP frozen. Their LR is
  `BASE_LR × CLIP_MULTIPLIER` (1e-4 × 0.01 = 1e-6; paper 2e-6); the swin backbone multiplier is 0.
- This copy of EVA-CLIP has none of stage 1's additions (SDPA, block-8 exit = placement A,
  placement B). Porting them is a later, explicit step.
- In-training eval: `DATASETS.TEST` = COCO-Stuff val (`coco_2017_test_stuff_all_sem_seg`) every
  `TEST.EVAL_PERIOD` 5000 iters, no sliding window; it picks `model_best.pth`. The six benchmarks
  use sliding window and pooling [1,1] (`eval.sh` / `scripts/exp_final_eval.sh`) on `model_final.pth`.
- `SOLVER.CHECKPOINT_PERIOD` 5000 (detectron2 default); tracked runs keep only the newest periodic file.
- Single-GPU fixes in the first commit: `torch.load(..., weights_only=False)` (4 files, torch ≥ 2.6),
  `xattn: false` in the EVA model configs, `IMS_PER_BATCH 1`, dataset/checkpoint paths, trimmed
  `requirements.txt` (mmcv/mmseg/xformers/detectron2-pin removed).

## Code style

Match the surrounding code: plain PyTorch + detectron2 conventions, short functions, few comments
(comment *why*), new options as config keys in `cat_seg/config.py` with defaults that keep upstream
behaviour unchanged. Repo conventions: `.claude/rules/repo-conventions.md`.
