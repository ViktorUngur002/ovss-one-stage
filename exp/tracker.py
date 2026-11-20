"""The tracker the training code talks to.

One commit = one wandb run, forever. Branch = wandb group. The training code only
ever calls `get_tracker()`, which returns a no-op object unless the run was started
through `run_experiment.py`, so launching the original way (`run.sh`, `eval.sh`,
`python train_net.py ...`) behaves exactly as upstream does.

Steps are detectron2 iterations (`global_step`); there are no epochs here.
"""
import logging
import time
from pathlib import Path

import torch

from . import paths, runmap, static, storage


class NullTracker:
    """What the training code gets when tracking is not active."""

    enabled = False

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None
        return _noop


_NULL = NullTracker()
_ACTIVE = None


def get_tracker():
    return _ACTIVE if _ACTIVE is not None else _NULL


def set_tracker(tracker):
    global _ACTIVE
    _ACTIVE = tracker


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _flatten(prefix, node, out):
    for key, value in node.items():
        if isinstance(value, dict):
            _flatten(f"{prefix}{key}.", value, out)
        else:
            out[f"{prefix}{key}"] = _jsonable(value)
    return out


def flatten_cfg(cfg):
    """detectron2 CfgNode -> {"SOLVER.BASE_LR": 0.0002, ...}"""
    import yaml
    return _flatten("", yaml.safe_load(cfg.dump()), {})


class Tracker:
    enabled = True

    def __init__(self, git_info, run_directory, argv, launch_script=None, job_type="train"):
        self.git = git_info
        self.job_type = job_type
        self.dir = Path(run_directory)
        self.argv = list(argv)
        self.launch_script = launch_script

        self.state = storage.read_state(self.dir)
        self.is_resume = bool(self.state)
        self._prev_train = float(self.state.get("train_seconds", 0.0))
        self._prev_eval = float(self.state.get("eval_seconds", 0.0))
        self.best_metric = self.state.get("best_metric")
        self.best_iter = self.state.get("best_iter")
        if self.state.get("best_metric_name", paths.BEST_METRIC_NAME) != paths.BEST_METRIC_NAME:
            # a best value stored under another metric is not comparable: rejudge from the next eval
            logging.warning("[exp] best metric changed %s -> %s, resetting best",
                            self.state["best_metric_name"], paths.BEST_METRIC_NAME)
            self.best_metric = self.best_iter = None
        self._global_step = int(self.state.get("global_step", 0))
        self._peak_memory_gb = float(self.state.get("peak_memory_gb", 0.0))

        self._t0 = time.time()
        self._eval_seconds = 0.0
        self._last_eval_best = False
        self._rclone = None
        self.run = None
        self._init_wandb()

    # ------------------------------------------------------------------ setup

    def _init_wandb(self):
        import wandb

        branch = self.git["branch"]
        # An eval of a commit is a separate run from training that commit, so it
        # gets its own key. One commit still means one run per job type.
        commit = self.git["commit"] + ("" if self.job_type == "train" else f":{self.job_type}")
        existing = runmap.lookup(branch, commit)
        tags = [f"model-slug:{self.git['slug'][:32]}"]
        if self.git["dirty"]:
            tags.append("dirty-tree")
        if self.git["detached"]:
            tags.append("detached-head")

        common = dict(project=paths.WANDB_PROJECT, entity=paths.WANDB_ENTITY, group=branch,
                      job_type=self.job_type)
        if existing:
            self.run = wandb.init(id=existing["run_id"], resume="allow", **common)
            logging.info("[exp] resumed wandb run %s (commit %s)", existing["run_id"], commit[:8])
        else:
            suffix = "" if self.job_type == "train" else f" [{self.job_type}]"
            self.run = wandb.init(
                name=(self.git["subject"] or self.git["slug"]) + suffix,
                notes=self.git["body"],
                tags=tags,
                **common,
            )
            runmap.record(branch, commit, self.run.id, self.git["slug"],
                          extra={"run_dir": str(self.dir), "run_url": self.run.url})
            logging.info("[exp] created wandb run %s (commit %s)", self.run.id, commit[:8])

        # Our own step axis, so a resume from the last periodic checkpoint (which
        # goes back a few hundred iterations) is not silently dropped by wandb.
        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")

        self.run.config.update(
            {f"git/{k}": _jsonable(v) for k, v in self.git.items()}, allow_val_change=True)
        self.run.config.update(
            {f"env/{k}": _jsonable(v) for k, v in static.env_info().items()}, allow_val_change=True)
        self.run.config.update({"exp/run_dir": str(self.dir)}, allow_val_change=True)

        if self.git["dirty"]:
            logging.warning(
                "[exp] WORKING TREE IS DIRTY (%d files) - this run is not reproducible "
                "from git history alone", self.git["dirty_file_count"])

    # ------------------------------------------------------------- static info

    def log_static(self, model, cfg):
        """Params and the full effective config. Called once, at start."""
        total, trainable = static.count_params(model)
        payload = {
            "model/params_total": total,
            "model/params_trainable": trainable,
            "model/params_total_M": round(total / 1e6, 2),
            "model/params_trainable_M": round(trainable / 1e6, 2),
        }
        cfg_dict = flatten_cfg(cfg)
        self.run.config.update(payload, allow_val_change=True)
        self.run.config.update({f"cfg/{k}": v for k, v in cfg_dict.items()}, allow_val_change=True)
        self.run.config.update({"env/rocm_smi": static.rocm_smi()}, allow_val_change=True)

        self._note_runtime_changes(cfg_dict)
        self.state["cfg"] = cfg_dict
        self._persist()
        self._copy_provenance()
        logging.info("[exp] %s params (%s trainable), run dir %s",
                     f"{total/1e6:.1f}M", f"{trainable/1e6:.1f}M", self.dir)

    def _copy_provenance(self):
        if self.launch_script:
            storage.copy_into(self.dir, self.launch_script)
        storage.write_json(self.dir / "config" / "argv.json",
                           {"argv": self.argv, "recorded_at": storage.utc_now()})

    def _note_runtime_changes(self, cfg_dict):
        """A crash fixed by changing a config value on the command line keeps the same run.

        The config panel would silently show only the new value, so every change
        is also recorded as a timestamped event, visible as a table in the run.
        """
        previous = self.state.get("cfg")
        if self.job_type != "train" or not (self.is_resume and previous):
            return                       # the eval job changes DATASETS.TEST per benchmark on purpose
        changed = {k: (previous.get(k), v) for k, v in cfg_dict.items() if previous.get(k) != v}
        if not changed:
            return
        events = self.state.setdefault("resume_events", [])
        for key, (old, new) in sorted(changed.items()):
            events.append({"at": storage.utc_now(), "global_step": self._global_step,
                           "key": key, "old": _jsonable(old), "new": _jsonable(new)})
            logging.warning("[exp] config change on resume: %s %r -> %r", key, old, new)
        self._log_resume_table(events)

    def _log_resume_table(self, events):
        try:
            import wandb
            table = wandb.Table(columns=["at", "global_step", "key", "old", "new"])
            for event in events:
                table.add_data(event["at"], event["global_step"], event["key"],
                               str(event["old"]), str(event["new"]))
            self.run.log({"resume_events": table, "global_step": self._global_step})
        except Exception as exc:
            logging.warning("[exp] could not log resume table: %s", exc)

    # ----------------------------------------------------------------- during

    def log_train(self, step, log_data):
        self._global_step = int(step)
        payload = {f"train/{k}": v for k, v in log_data.items()}
        payload["global_step"] = self._global_step
        payload.update(self._time_metrics())
        self.run.log(payload)

    def log_raw(self, step, data):
        """Scalars logged under their own keys (no train/ prefix), on the global_step axis."""
        payload = dict(data)
        payload["global_step"] = int(step)
        self.run.log(payload)

    def log_media(self, step, images):
        """Images (H, W, 3 uint8 arrays) on the same global_step axis as the training scalars."""
        import wandb
        payload = {k: wandb.Image(v) for k, v in images.items()}
        payload["global_step"] = int(step)
        self.run.log(payload)

    def log_eval(self, step, metrics, elapsed_seconds=0.0):
        """In-training evaluation (detectron2 EvalHook). Returns True if this is a new best."""
        self._global_step = int(step)
        self._eval_seconds += float(elapsed_seconds or 0.0)
        value = self._governing_metric(metrics)

        payload = {f"eval/{k}": v for k, v in metrics.items()}
        payload["global_step"] = self._global_step
        if value is not None:
            payload[f"eval/{paths.BEST_METRIC_NAME}"] = value
        payload.update(self._time_metrics())
        payload.update(self._memory_metrics())
        self.run.log(payload)

        is_best = value is not None and (self.best_metric is None or value > self.best_metric)
        if is_best:
            self.best_metric, self.best_iter = value, self._global_step
            self.run.summary[f"best_{paths.BEST_METRIC_NAME}"] = value
            self.run.summary["best_iter"] = self._global_step
            logging.info("[exp] new best %s = %.4f at iteration %s",
                         paths.BEST_METRIC_NAME, value, self._global_step)
        self._last_eval_best = is_best
        self._persist()
        return is_best

    def promote_best(self, state_dict, metrics=None):
        """After a new best: export fp16 weights and push them. detectron2's
        BestCheckpointer has already written the resumable model_best.pth."""
        if not self._last_eval_best:
            return False
        self._last_eval_best = False
        try:
            if state_dict is not None and paths.CLOUD_WEIGHTS_ENABLED:
                meta = {"commit": self.git["commit"], "branch": self.git["branch"],
                        "iteration": self.best_iter, paths.BEST_METRIC_NAME: self.best_metric,
                        "dtype": paths.CLOUD_WEIGHTS_DTYPE, "exported_at": storage.utc_now()}
                dest, size = storage.export_weights(state_dict, self.dir / "best_weights.pt", meta=meta)
                logging.info("[exp] best_weights.pt written (%.0f MB, %s)",
                             size / 2 ** 20, paths.CLOUD_WEIGHTS_DTYPE)
            storage.write_json(self.dir / "best_metrics.json", {
                paths.BEST_METRIC_NAME: self.best_metric,
                "best_metric_keys": list(paths.BEST_METRIC_KEYS),
                "iteration": self.best_iter,
                "date": storage.utc_now(),
                "commit": self.git["commit"],
                "branch": self.git["branch"],
                "metrics": _jsonable(metrics or {}),
            })
            self._persist()
            self._push_to_cloud()
            return True
        except Exception as exc:                 # never lose a run over bookkeeping
            logging.error("[exp] promote_best failed: %s: %s", type(exc).__name__, exc)
            return False

    def on_checkpoint(self, path, step):
        self._global_step = int(step)
        self.state.update({"latest_checkpoint": str(path), "global_step": int(step),
                           "latest_saved_at": storage.utc_now()})
        self._persist()

    def log_final_eval(self, dataset, metrics):
        """One benchmark of the post-training evaluation, logged to this (eval) run."""
        payload = {f"final/{dataset}/{k}": v for k, v in metrics.items()}
        payload["global_step"] = self._global_step
        self.run.log(payload)
        for key in ("mIoU", "mACC", "pACC", "fwIoU"):
            if key in metrics:
                self.run.summary[f"final_{key}/{dataset}"] = metrics[key]
        self.state.setdefault("final_eval", {})[dataset] = _jsonable(metrics)
        self._persist()

    # ------------------------------------------------------------------ helpers

    def _governing_metric(self, metrics):
        values = [metrics[k] for k in paths.BEST_METRIC_KEYS if k in metrics]
        if not values:
            return None
        return sum(values) / len(values)

    def _time_metrics(self):
        session = time.time() - self._t0
        train = self._prev_train + max(session - self._eval_seconds, 0.0)
        evaluation = self._prev_eval + self._eval_seconds
        return {"time/train_hours": train / 3600.0,
                "time/eval_hours": evaluation / 3600.0,
                "time/total_hours": (train + evaluation) / 3600.0}

    def _memory_metrics(self):
        if not torch.cuda.is_available():
            return {}
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        reserved = torch.cuda.max_memory_reserved() / 2 ** 30
        self._peak_memory_gb = max(self._peak_memory_gb, peak)
        self.run.summary["gpu/peak_memory_gb"] = self._peak_memory_gb
        torch.cuda.reset_peak_memory_stats()
        return {"gpu/peak_memory_gb_cycle": peak, "gpu/peak_reserved_gb_cycle": reserved}

    def _persist(self):
        times = self._time_metrics()
        self.state.update({
            "best_metric": self.best_metric,
            "best_iter": self.best_iter,
            "best_metric_name": paths.BEST_METRIC_NAME,
            "global_step": self._global_step,
            "train_seconds": times["time/train_hours"] * 3600.0,
            "eval_seconds": times["time/eval_hours"] * 3600.0,
            "peak_memory_gb": self._peak_memory_gb,
            "wandb_run_id": self.run.id if self.run else None,
            "commit": self.git["commit"],
            "branch": self.git["branch"],
            "job_type": self.job_type,
            "updated_at": storage.utc_now(),
        })
        try:
            storage.write_state(self.dir, self.state)
        except OSError as exc:
            logging.warning("[exp] could not write exp_state.json: %s", exc)

    def _push_to_cloud(self):
        if self._rclone is not None and self._rclone.poll() is None:
            logging.info("[exp] previous cloud push still running - skipping this one")
            return
        self._rclone = storage.rclone_push_async(
            self.dir,
            storage.remote_dir(self.git["branch_safe"], self.dir.name),
            ["best_weights.pt", "best_metrics.json", "exp_state.json",
             "config.yaml", "log.txt", "config/**"],
        )

    # ------------------------------------------------------------------- finish

    def finish(self, status="finished"):
        times = self._time_metrics()
        self.state["status"] = status
        self._persist()
        try:
            self.run.summary["status"] = status
            self.run.summary["time/train_hours"] = times["time/train_hours"]
            self.run.summary["time/eval_hours"] = times["time/eval_hours"]
            self.run.summary["time/total_hours"] = times["time/total_hours"]
        except Exception:
            pass

        if status == "finished" and self.job_type == "train":
            self._push_to_cloud()

        if self._rclone is not None:
            logging.info("[exp] waiting for the cloud push to finish...")
            try:
                _, err = self._rclone.communicate(timeout=3600)
                if self._rclone.returncode:
                    logging.warning("[exp] rclone exited %s: %s", self._rclone.returncode,
                                    (err or b"").decode(errors="replace")[:500])
                else:
                    logging.info("[exp] cloud push done")
            except Exception as exc:
                logging.warning("[exp] cloud push did not complete: %s", exc)

        try:
            self.run.finish(exit_code=0 if status == "finished" else 1)
        except Exception:
            pass
        logging.info("[exp] run finished (%s) - total %.2f h", status, times["time/total_hours"])
