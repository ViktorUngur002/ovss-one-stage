"""detectron2 glue for the tracker. Only imported when tracking is active.

- TrackerWriter: training scalars from detectron2's EventStorage (total_loss, loss_*,
  lr, time, data_time) to wandb under train/*, at the PeriodicWriter's period.
- attach(): wraps the EvalHook's function so every in-training evaluation goes to
  wandb under eval/*, saves model_best.pth + fp16 best_weights.pt on a new best, and
  limits the periodic checkpoints kept on disk. Nothing in the training math changes.
"""
import logging
import time

import torch
from detectron2.engine import hooks
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils.events import EventWriter, get_event_storage

from . import paths
from .tracker import get_tracker

# cfg.DATASETS.TEST name -> short name used in wandb keys and folders
DATASET_SHORT = {
    "coco_2017_test_stuff_all_sem_seg": "coco_stuff",
    "ade20k_150_test_sem_seg": "ade150",
    "ade20k_full_sem_seg_freq_val_all": "ade847",
    "voc_2012_test_sem_seg": "voc20",
    "voc_2012_test_background_sem_seg": "voc21",
    "context_59_test_sem_seg": "pc59",
    "context_459_test_sem_seg": "pc459",
}


def short_name(dataset):
    return DATASET_SHORT.get(dataset, dataset)


def split_metrics(results):
    """Flatten detectron2 results; per-class entries (IoU-<class>, ACC-<class>) go apart."""
    flat = {k: float(v) for k, v in flatten_results_dict(results).items()}
    summary = {k: v for k, v in flat.items() if "-" not in k.rsplit("/", 1)[-1]}
    per_class = {k: v for k, v in flat.items() if k not in summary}
    return summary, per_class


class TrackerWriter(EventWriter):
    def __init__(self, window_size=20):
        self._window = window_size

    def write(self):
        storage = get_event_storage()
        it = storage.iter
        if paths.LOG_EVERY_N_ITERS > 1 and (it + 1) % paths.LOG_EVERY_N_ITERS:
            return
        data = {}
        for name, (value, step) in storage.latest_with_smoothing_hint(self._window).items():
            # eval results are logged by the eval hook; only this iteration's training scalars
            if step == it and "/" not in name:
                data[name] = value
        if data:
            get_tracker().log_train(it + 1, data)


def attach(trainer, hook_list):
    """Called from Trainer.build_hooks() when tracking is active."""
    tracker = get_tracker()
    for hook in hook_list:
        if isinstance(hook, hooks.PeriodicCheckpointer) and paths.KEEP_PERIODIC_CHECKPOINTS > 0:
            hook.max_to_keep = paths.KEEP_PERIODIC_CHECKPOINTS
        if isinstance(hook, hooks.EvalHook):
            hook._func = _wrap_eval(trainer, hook._func)
    tracker.log_static(trainer.model, trainer.cfg)


def log_final_results(cfg, model, results):
    """--eval-only through run_experiment.py: one benchmark per launch (scripts/exp_final_eval.sh).
    Logged to the eval run and attached as final_mIoU/<dataset> to the training run."""
    tracker = get_tracker()
    tracker.log_static(model, cfg)
    datasets = list(cfg.DATASETS.TEST)
    # DefaultTrainer.test() unwraps the dict when there is a single dataset
    per_dataset = {datasets[0]: results} if len(datasets) == 1 else results
    for dataset, res in per_dataset.items():
        name = short_name(dataset)
        summary, _ = split_metrics(res)
        metrics = {k.split("/", 1)[-1]: v for k, v in summary.items()}
        tracker.log_final_eval(name, metrics)
        if "mIoU" in metrics:
            try:
                from .attach_final_eval import attach
                run_id, _ = attach(tracker.git["branch"], tracker.git["commit"], {name: metrics["mIoU"]})
                logging.info("[exp] %s mIoU %.2f attached to training run %s", name, metrics["mIoU"], run_id)
            except Exception as exc:
                logging.warning("[exp] could not attach %s to the training run: %s", name, exc)


def _wrap_eval(trainer, func):
    def tracked_eval():
        t0 = time.time()
        results = func()
        if not results:
            return results
        step = trainer.iter + 1
        summary, per_class = split_metrics(results)
        tracker = get_tracker()
        is_best = tracker.log_eval(step, summary, time.time() - t0)
        if per_class:
            tracker.log_raw(step, {f"eval_class/{k}": v for k, v in per_class.items()})
        if is_best:
            try:
                trainer.checkpointer.save("model_best", iteration=trainer.iter)
            except Exception as exc:
                logging.error("[exp] could not save model_best.pth: %s", exc)
            model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            with torch.no_grad():
                tracker.promote_best(model.state_dict(), summary)
        return results
    return tracked_eval
