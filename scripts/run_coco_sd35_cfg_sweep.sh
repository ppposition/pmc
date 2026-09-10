#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

UV=${UV:-uv}
GPU=${GPU:-0}
DATA_DIR=${DATA_DIR:-coco_data}
OUTPUT_ROOT=${OUTPUT_ROOT:-coco_data/sd35_coco30k}
MODEL_ID=${MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}
CFG_SCALES=${CFG_SCALES:-"4.5 5.0 6.0 7.0"}
NUM_SAMPLES=${NUM_SAMPLES:-30000}
SELECTION_SEED=${SELECTION_SEED:-42}
NOISE_SEED=${NOISE_SEED:-10000}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-512}
NUM_STEPS=${NUM_STEPS:-20}
SCHEDULER=${SCHEDULER:-euler}
CFG_SCHEDULE=${CFG_SCHEDULE:-norm-clamped}
GAMMA=${GAMMA:-1.2}

PROMPT_FILE="${DATA_DIR}/eval_manifests/coco_val_first_caption_n${NUM_SAMPLES}_seed${SELECTION_SEED}.txt"
REF_DIR="${DATA_DIR}/val2014"
PROTOCOL="coco2014-val-${NUM_SAMPLES}-unique-images-first-caption-seed${SELECTION_SEED}"
if [[ "$NUM_SAMPLES" == 30000 && "$SELECTION_SEED" == 42 ]]; then
    PROTOCOL=coco2014-val-30k-unique-images-first-caption-seed42
fi

for cfg in ${CFG_SCALES}; do
    # gamma only affects norm-clamped, so keep it out of the constant run name
    if [ "${CFG_SCHEDULE}" = "constant" ]; then
        run_name="${SCHEDULER}_${CFG_SCHEDULE}_cfg${cfg}_steps${NUM_STEPS}_${HEIGHT}x${WIDTH}"
    else
        run_name="${SCHEDULER}_${CFG_SCHEDULE}_cfg${cfg}_gamma${GAMMA}_steps${NUM_STEPS}_${HEIGHT}x${WIDTH}"
    fi
    gen_dir="${OUTPUT_ROOT}/${run_name}"
    result_file="${OUTPUT_ROOT}/metrics_${run_name}.txt"

    generation_args=(
        --data-dir "${DATA_DIR}"
        --out-dir "${gen_dir}"
        --model-id "${MODEL_ID}"
        --num-samples "${NUM_SAMPLES}"
        --selection-seed "${SELECTION_SEED}"
        --noise-seed "${NOISE_SEED}"
        --batch-size "${BATCH_SIZE}"
        --height "${HEIGHT}"
        --width "${WIDTH}"
        --num-steps "${NUM_STEPS}"
        --cfg-scale "${cfg}"
        --cfg-schedule "${CFG_SCHEDULE}"
        --scheduler "${SCHEDULER}"
        --gpu "${GPU}"
    )
    if [ "${CFG_SCHEDULE}" != "constant" ]; then
        generation_args+=(--cfg-gamma "${GAMMA}")
    fi

    "${UV}" run --locked python -m cfg_norm_clamped_eval.generate_coco_sd35 "${generation_args[@]}"

    # Modern reproducible COCO-30K suite: Clean-FID + CLIP ViT-L/14 raw
    # cosine + ImageReward-v1.0. Non-standard/secondary metrics are skipped.
    EVAL_GPU="${GPU}" "${UV}" run --locked python -m cfg_norm_clamped_eval.eval_metrics \
        --gen-dir "${gen_dir}" \
        --ref-dir "${REF_DIR}" \
        --prompt-file "${PROMPT_FILE}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --gpu "${GPU}" \
        --clip-model vitl14 \
        --fid-mode clean \
        --protocol "$PROTOCOL" \
        --skip-sfid \
        --skip-is \
        --skip-kid \
        --skip-cmmd \
        --save-results "${result_file}"
done
