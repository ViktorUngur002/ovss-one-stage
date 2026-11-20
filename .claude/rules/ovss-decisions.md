# OVSS stage — decisions, open parameters, known issues

Living record. Update it when the user settles something (date it), so later sessions don't re-open
closed questions or silently assume an undecided value.

## Decided

- **2026-11-20** Repo = the authors' `DeCLIP_CATSeg` branch (`bb2990c`) as one commit, plus the
  single-GPU fixes already made on the PC (torch.load `weights_only=False`, `xattn: false`,
  `IMS_PER_BATCH 1`, dataset/checkpoint paths, trimmed requirements). Upstream git history not kept.
  Branch `Initial_run_DeCLIP_CATSeg_results_reproduction`.
- **2026-11-20** First goal: reproduce the authors' CAT-Seg + DeCLIP EVA-L/14-336 result with their
  released backbone (`DeCLIP_EVA-L_DINOv2-L_csa_0.05_2.0.pt`, csa mode). Stage-1 code (qq mode,
  SDPA, placements A/B) is **not** ported yet.
- **2026-11-20** Evaluation as upstream: in-training eval on `DATASETS.TEST` (COCO-Stuff val) every
  `TEST.EVAL_PERIOD` (5000); the six benchmarks (ADE847, PC459, ADE150, PC59, VOC20, VOC21) only after
  training (`scripts/exp_final_eval.sh`, `model_final.pth` as upstream `eval.sh`). Best checkpoint =
  in-training COCO-Stuff mIoU.
- **2026-11-20** Tracking as in clip-JEPA-fine-tune: branch = wandb group, commit = run; wandb
  project `ovss-catseg`; runs in `/mnt/win-data/MasterDisertatie/MyRuns-OVSS/`; keep the newest
  periodic checkpoint + `model_best.pth` + `model_final.pth`; fp16 best weights to `gdrive:MyRuns-OVSS`.
- **2026-11-20** Environment: the existing conda env `DeCLIP_CATSeg`; launch scripts put this repo's
  `cat_seg/src` first on `PYTHONPATH` (the env's editable `open_clip` still points at
  `/home/viktor/MasterDisertatie/DeCLIP-CATSeg`).

## Undecided (do not assume a value; propose one with reasoning when we get there)

| Parameter | Upstream (paper, 4 GPUs) | Here now | Notes |
|---|---|---|---|
| Batch | 4 | 1 (`IMS_PER_BATCH`) | 20 GB; batch 2 untested |
| BASE_LR | 2e-4 | 2e-4 | linear/sqrt scaling to batch 1 not decided |
| MAX_ITER | 80000 | 80000 | at batch 1 = 1/4 of the paper's images seen |
| CLIP_MULTIPLIER | 0.01 | 0.01 | CLIP q/v LR = 2e-6 |
| EVAL_PERIOD | 5000 | 5000 | 16 COCO-Stuff evals over 80k iters |
| Final-eval weights | model_final | model_final | model_best is the alternative |
| Our backbones (TN e1, TW e1) and placements A/B in CAT-Seg | — | — | needs qq mode + stage-1 reading code ported |
