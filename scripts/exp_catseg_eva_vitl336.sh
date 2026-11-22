#!/usr/bin/env bash
# Tracked CAT-Seg training: reproduction of the authors' CAT-Seg + DeCLIP EVA-L/14-336 result
# (README: ADE847 17.96, PC459 25.89, ADE150 40.59, PC59 63.79, VOC20 97.76, VOC21 83.67).
#
#   * backbone: the authors' released DeCLIP checkpoint DeCLIP_EVA-L_DINOv2-L_csa_0.05_2.0 (MODEL.SEM_SEG_HEAD.CACHE_DIR
#     in configs/eva_vitl_336.yaml), dense features read in csa mode as in the upstream cat_seg_model.py.
#   * data: train COCO-Stuff (171 classes), in-training eval on COCO-Stuff val every TEST.EVAL_PERIOD (5000 iters),
#     the upstream defaults, run under fp16 autocast (TEST.AMP_DTYPE fp16 below; fp16 gave the same ADE150 mIoU as
#     fp32 and is ~2.4x faster). The six OVSS benchmarks run after training: scripts/exp_final_eval.sh (fp32 by default).
#   * single GPU, the paper's batch 4 exactly: SOLVER.IMS_PER_BATCH 1 x SOLVER.GRAD_ACCUM_STEPS 4 under fp16 AMP
#     (SOLVER.AMP.ENABLED), paper's SOLVER.BASE_LR 2e-4, CLIP q/v lr x0.01 = 2e-6, MAX_ITER 80000 optimizer steps =
#     320k images as in the paper. Batch 2 and 4 do not fit in 20 GB even with AMP (the CAT-Seg head's per-class
#     activations); AMP batch 1 runs at 0.54 s per micro-batch (fp32: 1.64 s), so ~48 h of training.
#   * output: MyRuns-OVSS/<branch>/<commit8>_<slug>/ (set by run_experiment.py); relaunching resumes from the last
#     periodic checkpoint there. Override any config value by appending KEY VALUE pairs to this script's arguments.
#
# Run from the DeCLIP_CATSeg conda env:  conda activate DeCLIP_CATSeg && bash scripts/exp_catseg_eva_vitl336.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
[ -f exp/env.sh ] && source exp/env.sh
export EXP_LAUNCH_SCRIPT="${BASH_SOURCE[0]}"
# this repo's copy of open_clip (CAT-Seg's EVA-CLIP) before the env's editable install
export PYTHONPATH="${REPO_ROOT}/cat_seg/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ "${CONDA_DEFAULT_ENV:-}" = "DeCLIP_CATSeg" ] || echo "[exp] WARNING: conda env is '${CONDA_DEFAULT_ENV:-none}', expected DeCLIP_CATSeg"

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} python run_experiment.py \
--config-file configs/eva_vitl_336.yaml \
--num-gpus 1 \
--dist-url auto \
TEST.AMP_DTYPE fp16 \
"$@"
