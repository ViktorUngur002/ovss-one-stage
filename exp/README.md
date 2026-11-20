# Experiment tracking (`exp/`)

Ported from clip-JEPA-fine-tune and adapted to detectron2. Git is the only bookkeeping:

* **branch = wandb group** (and the folder `MyRuns-OVSS/<branch>/`)
* **commit = one wandb run**, forever: subject = run name and folder slug, body = run notes
* the eval-only job of a commit is its own run, named `<subject> [eval]`

Write the commit so it reads as a label; put the *why* in the body.

## Launching

```bash
conda activate DeCLIP_CATSeg
bash scripts/exp_catseg_eva_vitl336.sh                 # tracked training
bash scripts/exp_final_eval.sh                         # the six OVSS benchmarks, after training
```

Both put this repo's `cat_seg/src` first on `PYTHONPATH`, so CAT-Seg's `open_clip` copy comes from
this repo, not from the env's editable install (the banner prints which `open_clip` was loaded, and
the run config records it as `env/open_clip_file`). Any config value can be overridden by appending
`KEY VALUE` pairs, e.g. `bash scripts/exp_catseg_eva_vitl336.sh SOLVER.BASE_LR 0.0001`.

The upstream `run.sh`, `eval.sh` and `python train_net.py ...` still work and behave exactly as
upstream: every hook is a no-op unless `run_experiment.py` started the run.

## Where things land

```
/mnt/win-data/MasterDisertatie/MyRuns-OVSS/<branch>/<commit8>_<slug>/     (= cfg.OUTPUT_DIR)
    model_XXXXXXX.pth  last_checkpoint   newest periodic checkpoint, used by --resume
    model_best.pth                        best by the in-training eval (resumable)
    model_final.pth                       end of training
    best_weights.pt                       weights only, fp16 - the cloud copy
    best_metrics.json  exp_state.json
    config.yaml  log.txt  metrics.json    written by detectron2
    config/                               launch script, argv
/mnt/win-data/MasterDisertatie/MyRuns-OVSS/<branch>/<commit8>_<slug>__eval/<benchmark>/
```

Google Drive (`gdrive:MyRuns-OVSS/...`) receives `best_weights.pt` and the small text files, in the
background. The resumable checkpoints stay local.

## What goes to wandb

| Key | When | What |
|---|---|---|
| `train/total_loss`, `train/loss_*`, `train/lr`, `train/time`, `train/data_time` | every 20 iters | detectron2's EventStorage, losses median-smoothed over 20 iters (as the console) |
| `eval/sem_seg/mIoU`, `fwIoU`, `mACC`, `pACC`, `eval/intrain_mIoU` | every `TEST.EVAL_PERIOD` and at the end | in-training eval on `DATASETS.TEST` (COCO-Stuff val upstream) |
| `eval_class/sem_seg/IoU-<class>`, `ACC-<class>` | same | per-class numbers |
| `gpu/peak_memory_gb_cycle`, `time/*_hours` | at each eval | memory since the last eval, wall time split train/eval |
| `final/<benchmark>/*` (eval run), `final_mIoU/<benchmark>` (training run summary) | after training | the six OVSS benchmarks |
| config `cfg/*`, `git/*`, `env/*`, `model/params_*` | start | full effective config, git state, machine |

## What is "best"

The mean of `EXP_BEST_METRIC_KEYS` (default `sem_seg/mIoU`) from the in-training evaluation, i.e.
COCO-Stuff val mIoU with the upstream config. A new best saves `model_best.pth` and exports
`best_weights.pt`. The six OVSS benchmarks are attached afterwards and never pick the checkpoint;
`scripts/exp_final_eval.sh` evaluates `model_final.pth` by default, as upstream `eval.sh` does.

## Crash recovery

Relaunch the same script. `run_experiment.py` always passes `--resume` for training, so detectron2
continues from `last_checkpoint` in the run folder (up to `SOLVER.CHECKPOINT_PERIOD` iterations
are redone), on the same wandb run and step axis. A config value changed on the command line
without committing is recorded as a row in the run's `resume_events` table. Changed code? Commit
it - that is a new run in the same group, by design.

## Configuration

Machine-specific settings are in `exp/env.sh` (untracked, copied from `exp/env.sh.example`): wandb
project/entity, `EXP_MYRUNS_ROOT`, `DETECTRON2_DATASETS`, `WANDB_DIR`, `TORCH_HOME`, the rclone
remote, fp16/fp32 cloud weights, how many periodic checkpoints to keep, the wandb logging period.

Numbers that come from elsewhere can be attached to a training run by hand:

```bash
python -m exp.attach_final_eval --result ade150=40.6 --result pc59=63.8
python -m exp.where      # prints the training run folder of HEAD
```
