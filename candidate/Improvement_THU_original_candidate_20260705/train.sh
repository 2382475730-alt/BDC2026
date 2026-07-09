#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

cd "$APP_DIR"
python "$APP_DIR/code/src/train.py" "$@"
