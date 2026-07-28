#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$APP_DIR/code/src" "$APP_DIR/data" "$APP_DIR/model" "$APP_DIR/output" "$APP_DIR/temp"
test -f "$APP_DIR/data/train.csv"
test -f "$APP_DIR/data/test.csv"
test -f "$APP_DIR/model/config.json"

echo "Candidate-D initialized at $APP_DIR"
