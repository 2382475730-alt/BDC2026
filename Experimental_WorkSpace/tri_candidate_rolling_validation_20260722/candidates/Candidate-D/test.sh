#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONHASHSEED=20260708

cd "$APP_DIR"
python "$APP_DIR/code/src/test.py" \
  --work-dir "$APP_DIR" \
  --history "data/train.csv" \
  --test-data "data/test.csv" \
  --output "output/result.csv" \
  "$@"
