#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
source "$REPO_ROOT/src/frontier/common.sh"
check_repo_layout

for f in 1244.pdf 347.pdf 393.pdf 459.pdf 605.pdf; do
  [[ -f "$PAPERS_DIR/$f" ]] || { echo "Missing $PAPERS_DIR/$f" >&2; exit 2; }
done
[[ -f "$HUMAN_DIR/reviewer_1.csv" ]] || { echo "Missing $HUMAN_DIR/reviewer_1.csv" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH/config.json" >&2; exit 2; }
[[ -f "$VLLM_SIF" ]] || { echo "Missing vLLM container: $VLLM_SIF" >&2; exit 2; }
[[ -d "$CLIENT_ENV" ]] || { echo "Missing client environment: $CLIENT_ENV" >&2; exit 2; }

echo "Repository: $REPO_ROOT"
echo "Model:      $MODEL_PATH"
echo "Container:  $VLLM_SIF"
echo "Client env: $CLIENT_ENV"
echo "Papers:     $PAPERS_DIR"
echo "Human:      $HUMAN_DIR/reviewer_1.csv"
echo
python - <<PY
import csv, json
from pathlib import Path
cfg = json.loads(Path("$MODEL_PATH/config.json").read_text())
print("Model type:", cfg.get("model_type"))
print("Architectures:", cfg.get("architectures"))
print("Configured max positions:", cfg.get("max_position_embeddings", cfg.get("text_config", {}).get("max_position_embeddings")))
with open("$HUMAN_DIR/reviewer_1.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
print("Human reviewer rows:", len(rows))
print("Covidence IDs:", [r.get("Covidence #") for r in rows])
PY

echo
sha256sum "$HUMAN_DIR/reviewer_1.csv" "$MODEL_PATH/config.json" "$MODEL_PATH/model.safetensors.index.json"
