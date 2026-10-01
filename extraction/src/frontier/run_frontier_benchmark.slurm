#!/bin/bash
#SBATCH -A csc677
#SBATCH -J glimmer_extract
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o glimmer_extract-%j.out
#SBATCH -e glimmer_extract-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
source "$REPO_ROOT/src/frontier/common.sh"
check_repo_layout

for f in 1244.pdf 347.pdf 393.pdf 459.pdf 605.pdf; do
  [[ -f "$PAPERS_DIR/$f" ]] || { echo "Missing $PAPERS_DIR/$f" >&2; exit 2; }
done
[[ -f "$HUMAN_DIR/reviewer_1.csv" ]] || { echo "Missing $HUMAN_DIR/reviewer_1.csv" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
[[ -f "$VLLM_SIF" ]] || { echo "Missing vLLM container: $VLLM_SIF" >&2; exit 2; }
[[ -d "$CLIENT_ENV" ]] || { echo "Missing client env: $CLIENT_ENV" >&2; exit 2; }

load_frontier_gpu_runtime
GPU_LIST="$(visible_gpu_list)"
export ROCR_VISIBLE_DEVICES="$GPU_LIST"
export HIP_VISIBLE_DEVICES="$GPU_LIST"

SERVER_LOG="$LOG_DIR/vllm-${SLURM_JOB_ID}.log"
cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM server PID $VLLM_PID"
    kill "$VLLM_PID" || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Repo: $REPO_ROOT"
echo "Model: $MODEL_PATH"
echo "Container: $VLLM_SIF"
echo "Tensor parallel: $TP_SIZE"
echo "Visible GCDs: $GPU_LIST"
echo "Max model len: $MAX_MODEL_LEN"
echo "Reasoning strength: $REASONING_STRENGTH"

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

echo "Waiting for vLLM API readiness..."
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
if [[ "$READY" != "1" ]]; then
  echo "vLLM did not become ready. Last 150 log lines:" >&2
  tail -n 150 "$SERVER_LOG" >&2 || true
  exit 3
fi

echo "vLLM ready."
activate_client_env
export GLIMMER_BASE_URL="http://127.0.0.1:${PORT}/v1"
export GLIMMER_API_KEY="EMPTY"
export GLIMMER_MODEL="Muse-Glimmer-30B"
export GLIMMER_JSON_MODE="1"
export GLIMMER_SERIAL="0"
export GLIMMER_RETRIES="2"

# Reproducibility metadata.
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "repo_root=$REPO_ROOT"
  echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "model_path=$MODEL_PATH"
  echo "vllm_sif=$VLLM_SIF"
  echo "tp_size=$TP_SIZE"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "reasoning_strength=$REASONING_STRENGTH"
  echo "human_sha256=$(sha256sum "$HUMAN_DIR/reviewer_1.csv" | awk '{print $1}')"
  echo "model_config_sha256=$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')"
  echo "model_index_sha256=$(sha256sum "$MODEL_PATH/model.safetensors.index.json" | awk '{print $1}')"
  echo -n "vllm_version="
  apptainer exec "$VLLM_SIF" python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown
} > "$RESULTS_DIR/run_metadata.txt"

# Convert the completed Covidence reviewer export into the internal reference schema.
python "$SRC_DIR/prepare_human_reference.py" \
  --input "$HUMAN_DIR/reviewer_1.csv" \
  --output "$HUMAN_DIR/human_reference.csv"

# S0 + S1 + M1 on all five papers. --resume makes a requeued/restarted job reuse
# completed per-paper JSON files rather than spending GPU time again.
python "$SRC_DIR/run_extraction.py" \
  --papers "$PAPERS_DIR" \
  --out "$RESULTS_DIR" \
  --mode all \
  --resume \
  --base-url "$GLIMMER_BASE_URL" \
  --model "$GLIMMER_MODEL" \
  --temperature 0.0 \
  --max-tokens 6000

python "$SRC_DIR/compare_human.py" \
  --model "$RESULTS_DIR/model_extractions.csv" \
  --human "$HUMAN_DIR/human_reference.csv" \
  --out "$RESULTS_DIR/comparison_summary.csv" \
  --details "$RESULTS_DIR/comparison_by_field.csv" \
  --by-paper "$RESULTS_DIR/comparison_by_paper.csv"

python "$SRC_DIR/make_manual_review.py" \
  --model "$RESULTS_DIR/model_extractions.csv" \
  --human "$HUMAN_DIR/human_reference.csv" \
  --out "$RESULTS_DIR/manual_review.csv"

# This table is immediately useful for automatic agreement/cost metrics. Human
# field accuracy remains blank until manual_review.csv is semantically judged.
python "$SRC_DIR/benchmark_report.py" \
  --comparison "$RESULTS_DIR/comparison_summary.csv" \
  --model "$RESULTS_DIR/model_extractions.csv" \
  --manual "$RESULTS_DIR/manual_accuracy_summary.csv" \
  --out "$RESULTS_DIR/benchmark_table.csv" \
  --markdown "$RESULTS_DIR/benchmark_table.md"

echo
echo "========== AUTOMATIC COMPARISON =========="
column -s, -t < "$RESULTS_DIR/comparison_summary.csv" 2>/dev/null || cat "$RESULTS_DIR/comparison_summary.csv"
echo
echo "========== BENCHMARK TABLE =========="
cat "$RESULTS_DIR/benchmark_table.md"
echo
echo "Results: $RESULTS_DIR"
echo "Server log: $SERVER_LOG"
