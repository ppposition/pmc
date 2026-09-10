#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
"${UV:-uv}" run --locked python -m cfg_norm_clamped_eval.generate_imagenet "$@"
