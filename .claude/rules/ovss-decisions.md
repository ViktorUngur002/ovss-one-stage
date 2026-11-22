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
  project `OVSS One stage arch experiments`; runs in `/mnt/win-data/MasterDisertatie/MyRuns-OVSS/`; keep the newest
  periodic checkpoint + `model_best.pth` + `model_final.pth`; fp16 best weights to `gdrive:MyRuns-OVSS`.
- **2026-11-20** Environment: the existing conda env `DeCLIP_CATSeg`; launch scripts put this repo's
  `cat_seg/src` first on `PYTHONPATH` (the env's editable `open_clip` still points at
  `/home/viktor/MasterDisertatie/DeCLIP-CATSeg`).
- **2026-11-20** `SOLVER.BASE_LR` **1e-4** (configs/eva_vitl_336.yaml) for batch 1: the paper's 2e-4 at
  batch 4 scaled by sqrt(1/4), the square-root rule for Adam/AdamW (Malladi et al. 2022). Rejected:
  keeping 2e-4 (full-model clip norm 0.01 already fixes the step size, but each step is 4x noisier in
  direction) and linear scaling to 5e-5 (stage 1's rule, derived for SGD, likely too conservative for
  AdamW at a fixed 80k iterations). CLIP q/v lr follows at x0.01 = 1e-6.

- **2026-11-22** Run 3cb3861 (batch 1, lr 1e-4) finished: best = iteration 70000 (COCO-Stuff val 47.38);
  fp16 sliding-window eval (`MyRuns-OVSS/evals/01_*`): ADE150 38.75, PC59 61.57 vs the authors' 40.59 / 63.79.
  Next run (user's choice): `IMS_PER_BATCH` **2**, `BASE_LR` back to the paper's **2e-4** (unscaled; the sqrt
  rule would give 1.41e-4). Two changes at once: twice the images seen at 80k iterations and a higher lr, so
  the run cannot separate them. Batch 1 trained at 16.5 GB peak, so batch 2 may not fit in 20 GB.
  Eval-only `TEST.AMP_DTYPE` ("" = fp32, "fp16", "bf16") added; training unaffected.

- **2026-11-22** Batch-2 run f4a1d28 OOM'd at iteration 0 (19.1 GB). fp32 eval of 3cb3861 model_best = 38.75 ADE150,
  identical to fp16 (38.75): fp16 eval is safe. Eval 02 (`MyRuns-OVSS/evals/02_*`, fp32): the **authors' released
  CAT-Seg checkpoint gives 40.59 ADE150 on this machine** (= their README) and our model_final 38.80, so eval, data,
  ROCm and code are cleared; the gap comes from training (3cb3861 saw 80k images vs the paper's 320k, at half the lr).
- **2026-11-22** AMP memory test (100 iters, `MyRuns-OVSS/scratch/amp_memtest`): AMP batch 1 = 14.2 GB, 0.541 s/iter
  (fp32 1.636 s); AMP batch 2 and 4 OOM at iteration 0 in the CAT-Seg head (swin MLP, per-class activations).
  Next run (user's choice): the paper's batch 4 exactly as **IMS_PER_BATCH 1 x SOLVER.GRAD_ACCUM_STEPS 4** under
  **fp16 AMP** (`SOLVER.AMP.ENABLED`), `BASE_LR` 2e-4, MAX_ITER 80000 optimizer steps (320k images), ~48 h training.
  In-training evals under fp16 (`TEST.AMP_DTYPE` now applies to them too, set in the launch script); final
  benchmarks stay fp32 by default. Only difference from the authors' recipe: fp16 numerics. SDPA in the EVA
  backbone / swin window attention kept in reserve (might fit real batch 2, cannot fit batch 4).
- **Known upstream quirk (not fixed):** `CATSegPredictor.get_text_embeds` caches the text embeddings at the first
  eval and reuses them in every later in-training eval although the text encoder keeps training, so in-training
  COCO-Stuff numbers after the first eval (and the model_best pick) use stale text features. Final benchmarks
  (fresh model) are unaffected.

## Undecided (do not assume a value; propose one with reasoning when we get there)

| Parameter | Upstream (paper, 4 GPUs) | Here now | Notes |
|---|---|---|---|
| Batch | 4 | 1 x 4 accumulated (AMP) | decided 2026-11-22; batch 2+ does not fit even with AMP |
| MAX_ITER | 80000 | 80000 | optimizer steps; with accumulation = the paper's 320k images |
| CLIP_MULTIPLIER | 0.01 | 0.01 | CLIP q/v LR = 2e-6 with BASE_LR 2e-4 |
| EVAL_PERIOD | 5000 | 5000 | 16 COCO-Stuff evals; fp16 ~12 min each (fp32 28.5 min) |
| Final-eval weights | model_final | model_final | model_best is the alternative |
| Our backbones (TN e1, TW e1) and placements A/B in CAT-Seg | — | — | needs qq mode + stage-1 reading code ported |
