# WandB monitoring — how to add it

Workflow (from CLAUDE.md): after a piece is implemented and reviewed, offer monitoring options in
plain language; the user chooses; only then hand over (or write, if asked) the logging code.

## Plumbing that already exists

- Training code only talks to `exp.tracker.get_tracker()`. Unless the run was launched through
  `run_experiment.py` it is a no-op object, so **every logging call must be safe on the null
  tracker** (call tracker methods; never import `wandb` in model or training code).
- Training scalars: detectron2's `EventStorage`. Anything put there with
  `get_event_storage().put_scalar(name, value)` inside the model's forward or a hook is picked up by
  `exp.d2_hooks.TrackerWriter` every 20 iterations and logged as `train/<name>` (median over 20 iters
  for smoothed scalars). Names without `/` only; eval results use `/` and are logged separately.
  Losses returned in the model's loss dict are summed into the total loss by detectron2 — never put
  diagnostics there; use `put_scalar` instead (inside `if self.training:`).
- In-training eval: `exp.d2_hooks._wrap_eval` logs detectron2's results as `eval/sem_seg/*` (summary)
  and `eval_class/sem_seg/*` (per class), picks the best by `EXP_BEST_METRIC_KEYS`, saves
  `model_best.pth` and `best_weights.pt`.
- Arbitrary scalars on the same step axis: `get_tracker().log_raw(step, {...})`; images (H×W×3 uint8):
  `get_tracker().log_media(step, {name: array})`. Step = detectron2 iteration (`trainer.iter + 1`).
- Final benchmarks: `exp.d2_hooks.log_final_results` → `final/<benchmark>/*` in the eval run and
  `final_mIoU/<benchmark>` on the training run.

## Rules for diagnostics

- Heavy diagnostics (feature PCA, cost-volume heatmaps, prediction overlays) run at a **separate, rare
  interval** (new config key, e.g. every few thousand iterations), on one fixed image when possible,
  under `torch.no_grad()`, detached and moved to CPU — never inside the autograd graph, never every step.
- Keep the scalars that detect failure early: loss, lr per param group, grad norm (CLIP group vs head),
  peak VRAM, data time vs step time.
- Name keys by component: `catseg/...`, `clip/...`, `head/...`, so wandb panels group cleanly.
- Log what the user can interpret: say in the option description what a healthy vs broken curve or
  image looks like.
