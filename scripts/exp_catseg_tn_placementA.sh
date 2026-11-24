#!/usr/bin/env bash
# Tracked CAT-Seg training on our stage-1 backbone TN instead of DeCLIP's, read with placement A.
#
#   * backbone: TN (clip-JEPA-fine-tune dd067b2b) epoch 6, the epoch with the highest COCO maskpool (63.4 with
#     placement A), MODEL.SEM_SEG_HEAD.CACHE_DIR in configs/eva_vitl_336_tn_placementA.yaml. Dense features in qq mode
#     with placement A: the last block averages its values by softmax(cos(g_i, g_j) / 0.07), g = the patch tokens after
#     block 8 (EXIT_BLOCK 8, EXIT_TAU 0.07), no gradient through that table. The last block's q_proj is therefore
#     unused and untrained (DeCLIP's csa exit lets CAT-Seg retrain it: +4.95% weight change in the authors' model).
#   * differs from the reproduction run d594d7e (DeCLIP backbone, csa, stopped at 55k: 40.04 ADE150 bf16) only in the
#     backbone and its dense exit. Recipe identical: COCO-Stuff, bf16 AMP, IMS_PER_BATCH 1 x GRAD_ACCUM_STEPS 4,
#     BASE_LR 2e-4, CLIP q/v lr x0.01, MAX_ITER 80000, in-training COCO-Stuff eval every 5000 iters under bf16.
#   * final benchmarks: bash scripts/exp_final_eval.sh configs/eva_vitl_336_tn_placementA.yaml (the same config, or the
#     eval reads the weights through csa).
#   * output: MyRuns-OVSS/<branch>/<commit8>_<slug>/ (set by run_experiment.py); relaunching resumes from the last
#     periodic checkpoint there. Override any config value by appending KEY VALUE pairs to this script's arguments.
#
# Run from the DeCLIP_CATSeg conda env:  conda activate DeCLIP_CATSeg && bash scripts/exp_catseg_tn_placementA.sh
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
--config-file configs/eva_vitl_336_tn_placementA.yaml \
--num-gpus 1 \
--dist-url auto \
TEST.AMP_DTYPE bf16 \
"$@"
