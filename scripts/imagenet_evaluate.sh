#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
"${UV:-uv}" run --locked --extra imagenet python -m cfg_norm_clamped_eval.eval_imagenet --gen-dir outputs/imagenet --batch \
    --device "${DEVICE:-cuda:0}" --batch-size 64 \
    --save outputs/imagenet/metrics.csv --save-txt outputs/imagenet/metrics.txt
