#!/usr/bin/env bash
# Tracked CAT-Seg training on our stage-1 backbone TW instead of DeCLIP's, read with placement A+B.
#
#   * backbone: TW (clip-JEPA-fine-tune 9c652ffa) epoch 1, the epoch with the best training-free ADE20K mIoU under
#     A+B (21.21; epoch 6: 19.61), MODEL.SEM_SEG_HEAD.CACHE_DIR in configs/eva_vitl_336_tw_placementAB.yaml.
#     Dense features in qq mode with placement A+B in training and evaluation, so the CAT-Seg head trains on the
#     features it is evaluated on: A = the last block averages its values by softmax(cos(g_i, g_j) / 0.07) (EXIT_BLOCK 8,
#     EXIT_TAU 0.07, how TW was trained); B = each finished dense feature becomes the softmax(cos(g_i, g_j) / 0.1)-weighted
#     mean of the unit-length features (POST_BLOCK 8, POST_TAU 0.1); g = patch tokens after block 8. Neither table gets a
#     gradient; B's averaging does, so CAT-Seg's loss still reaches CLIP. The last block's q_proj is unused (as in the TN run).
#   * differs from the TN run a7baae2 (TN epoch 6, placement A, 40.61 ADE150 bf16) in the backbone and placement B.
#     Recipe identical: COCO-Stuff, bf16 AMP, IMS_PER_BATCH 1 x GRAD_ACCUM_STEPS 4, BASE_LR 2e-4, CLIP q/v lr x0.01,
#     MAX_ITER 80000, in-training COCO-Stuff eval every 5000 iters under bf16.
#   * final benchmarks: bash scripts/exp_final_eval.sh configs/eva_vitl_336_tw_placementAB.yaml (the same config, or the
#     eval reads the weights through csa).
#   * output: MyRuns-OVSS/<branch>/<commit8>_<slug>/ (set by run_experiment.py); relaunching resumes from the last
#     periodic checkpoint there. Override any config value by appending KEY VALUE pairs to this script's arguments.
#
# Run from the DeCLIP_CATSeg conda env:  conda activate DeCLIP_CATSeg && bash scripts/exp_catseg_tw_placementAB.sh
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
--config-file configs/eva_vitl_336_tw_placementAB.yaml \
--num-gpus 1 \
--dist-url auto \
TEST.AMP_DTYPE bf16 \
"$@"
