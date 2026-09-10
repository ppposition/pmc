#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
UV=${UV:-uv}
GPU=${GPU:-0}
REF_DIR=${REF_DIR:-data/imagenet_val}
for method in constant norm_clamped tv_cfg; do
    guidance=()
    if [[ "$method" == norm_clamped ]]; then
        guidance=(--cfg-schedule norm_clamped --cfg-gamma 1.1)
    elif [[ "$method" == tv_cfg ]]; then
        guidance=(--cfg-schedule tv_cfg)
    fi
    "$UV" run --locked python -m cfg_norm_clamped_eval.eval_fid --gpu "$GPU" --cfg-scale 2.0 \
        "${guidance[@]}" --sampling-method euler --num-sampling-steps 50 \
        --num-samples 50000 --batch-size 64 --ref-dir "$REF_DIR" \
        --out-dir "outputs/imagenet/$method" --skip-diversity
done
