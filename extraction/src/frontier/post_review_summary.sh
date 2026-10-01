#!/bin/bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$PWD}"
source "$REPO_ROOT/src/frontier/common.sh"
activate_client_env

python "$SRC_DIR/summarize_manual_review.py" \
  --review "$RESULTS_DIR/manual_review.csv" \
  --out "$RESULTS_DIR/manual_accuracy_summary.csv"

python "$SRC_DIR/benchmark_report.py" \
  --comparison "$RESULTS_DIR/comparison_summary.csv" \
  --model "$RESULTS_DIR/model_extractions.csv" \
  --manual "$RESULTS_DIR/manual_accuracy_summary.csv" \
  --out "$RESULTS_DIR/benchmark_table.csv" \
  --markdown "$RESULTS_DIR/benchmark_table.md"

cat "$RESULTS_DIR/benchmark_table.md"
