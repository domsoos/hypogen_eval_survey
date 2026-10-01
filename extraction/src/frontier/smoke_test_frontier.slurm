#!/bin/bash
#SBATCH -A csc677
#SBATCH -J glimmer_smoke
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o glimmer_smoke-%j.out
#SBATCH -e glimmer_smoke-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
source "$REPO_ROOT/src/frontier/common.sh"
check_repo_layout

[[ -f "$PAPERS_DIR/1244.pdf" ]] || { echo "Missing $PAPERS_DIR/1244.pdf" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
[[ -f "$VLLM_SIF" ]] || { echo "Missing vLLM container: $VLLM_SIF" >&2; exit 2; }
[[ -d "$CLIENT_ENV" ]] || { echo "Missing client env: $CLIENT_ENV" >&2; exit 2; }

SMOKE_RESULTS="$REPO_ROOT/smoke_results"
rm -rf "$SMOKE_RESULTS"
mkdir -p "$SMOKE_RESULTS" "$LOG_DIR"

load_frontier_gpu_runtime
GPU_LIST="$(visible_gpu_list)"
export ROCR_VISIBLE_DEVICES="$GPU_LIST"
export HIP_VISIBLE_DEVICES="$GPU_LIST"

SERVER_LOG="$LOG_DIR/vllm-smoke-${SLURM_JOB_ID}.log"
cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Node: $(hostname)"
echo "Visible GCDs: $GPU_LIST"
echo "Starting Muse-Glimmer server..."

apptainer exec \
  --bind "$MODEL_PATH:$MODEL_PATH:ro" \
  --env HF_HOME="$HF_HOME" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  "$VLLM_SIF" \
  vllm serve "$MODEL_PATH" \
    --served-model-name Muse-Glimmer-30B \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --reasoning-parser muse_glimmer \
    --default-chat-template-kwargs "{\"reasoning_strength\":\"${REASONING_STRENGTH}\"}" \
    --disable-log-requests \
  >"$SERVER_LOG" 2>&1 &
VLLM_PID=$!

READY=0
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during startup. Last 150 log lines:" >&2
    tail -n 150 "$SERVER_LOG" >&2 || true
    exit 3
  fi
  sleep 5
done
[[ "$READY" == "1" ]] || { tail -n 150 "$SERVER_LOG" >&2 || true; exit 3; }

echo "vLLM ready."
activate_client_env
export GLIMMER_BASE_URL="http://127.0.0.1:${PORT}/v1"
export GLIMMER_API_KEY="EMPTY"
export GLIMMER_MODEL="Muse-Glimmer-30B"
export GLIMMER_JSON_MODE="1"
export GLIMMER_RETRIES="2"

python "$SRC_DIR/run_extraction.py" \
  --papers "$PAPERS_DIR" \
  --paper-ids 1244 \
  --out "$SMOKE_RESULTS" \
  --mode single \
  --base-url "$GLIMMER_BASE_URL" \
  --model "$GLIMMER_MODEL" \
  --temperature 0.0 \
  --max-tokens 6000

echo
echo "========== SMOKE OUTPUT =========="
cat "$SMOKE_RESULTS/model_extractions.csv"
echo
echo "Server log: $SERVER_LOG"
