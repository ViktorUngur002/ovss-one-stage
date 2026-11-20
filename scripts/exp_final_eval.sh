#!/usr/bin/env bash
# Tracked final evaluation on the six OVSS benchmarks, the same settings as the upstream eval.sh
# (sliding window, pooling [1,1], each benchmark's class list). One eval-only launch per benchmark; all six
# log into one wandb run "<subject> [eval]" and attach final_mIoU/<benchmark> to the training run of this commit.
#
#   usage: bash scripts/exp_final_eval.sh [CONFIG] [WEIGHTS] [KEY VALUE ...]
#     CONFIG   default configs/eva_vitl_336.yaml (must match the training run)
#     WEIGHTS  default <training run folder of HEAD>/model_final.pth (as upstream eval.sh);
#              pass .../model_best.pth to evaluate the best in-training checkpoint instead
#   Benchmarks: ADE150, ADE847, VOC20, VOC21 (voc20b), PC59, PC459. Order as in eval.sh.
#
# Run from the DeCLIP_CATSeg conda env, after training has finished on the same commit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
[ -f exp/env.sh ] && source exp/env.sh
export EXP_LAUNCH_SCRIPT="${BASH_SOURCE[0]}"
export PYTHONPATH="${REPO_ROOT}/cat_seg/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

config=${1:-configs/eva_vitl_336.yaml}
weights=${2:-"$(python -m exp.where)/model_final.pth"}
shift $(( $# > 2 ? 2 : $# ))
[ -f "${weights}" ] || { echo "[exp] weights not found: ${weights}"; exit 1; }
echo "[exp] config ${config}, weights ${weights}"

run() {   # $1 class json, $2 dataset name, then extra KEY VALUE pairs
  local json=$1 dataset=$2
  shift 2
  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} python run_experiment.py \
    --config-file "${config}" --num-gpus 1 --dist-url auto --eval-only \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${json}" \
    DATASETS.TEST "(\"${dataset}\",)" \
    TEST.SLIDING_WINDOW "True" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    MODEL.WEIGHTS "${weights}" \
    "$@"
}
shift_opts=("$@")

run datasets/ade150.json ade20k_150_test_sem_seg "${shift_opts[@]}"
run datasets/ade847.json ade20k_full_sem_seg_freq_val_all "${shift_opts[@]}"
run datasets/voc20.json voc_2012_test_sem_seg "${shift_opts[@]}"
run datasets/voc20b.json voc_2012_test_background_sem_seg "${shift_opts[@]}"
run datasets/pc59.json context_59_test_sem_seg "${shift_opts[@]}"
run datasets/pc459.json context_459_test_sem_seg "${shift_opts[@]}"
echo "[exp] all six benchmarks done"
