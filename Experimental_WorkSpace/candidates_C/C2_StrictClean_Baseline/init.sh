#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$APP_DIR/data" "$APP_DIR/model" "$APP_DIR/output" "$APP_DIR/temp"
mkdir -p "$APP_DIR/code/src"

test -f "$APP_DIR/data/train.csv"
test -f "$APP_DIR/data/test.csv"

echo "Submission app initialized at $APP_DIR"
