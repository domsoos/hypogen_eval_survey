#!/bin/bash
# Shared Frontier configuration. Source this file from the SLURM/setup scripts.

PROJECT_ID="${PROJECT_ID:-csc677}"
USER_NAME="${USER_NAME:-domsoos}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

SRC_DIR="${SRC_DIR:-$REPO_ROOT/src}"
PAPERS_DIR="${PAPERS_DIR:-$REPO_ROOT/papers}"
HUMAN_DIR="${HUMAN_DIR:-$REPO_ROOT/human}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

MODEL_PATH="${MODEL_PATH:-/lustre/orion/${PROJECT_ID}/scratch/${USER_NAME}/models/Muse-Glimmer-30B}"
CLIENT_ENV="${CLIENT_ENV:-/lustre/orion/${PROJECT_ID}/scratch/${USER_NAME}/envs/glimmer-extraction-client}"
CONTAINER_DIR="${CONTAINER_DIR:-/lustre/orion/${PROJECT_ID}/scratch/${USER_NAME}/containers}"
CACHE_DIR="${CACHE_DIR:-/lustre/orion/${PROJECT_ID}/scratch/${USER_NAME}/.cache/glimmer-vllm}"

# Muse-Glimmer support is present in current vLLM releases. Keep this pinned for
# reproducibility, but it can be overridden at submission/setup time.
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
VLLM_SIF="${VLLM_SIF:-${CONTAINER_DIR}/vllm-openai-rocm-v${VLLM_VERSION}.sif}"

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
REASONING_STRENGTH="${REASONING_STRENGTH:-medium}"
USE_OLCF_GPU_BIND="${USE_OLCF_GPU_BIND:-1}"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_DIR/triton}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

check_repo_layout() {
  [[ -f "$SRC_DIR/run_extraction.py" ]] || {
    echo "Expected $SRC_DIR/run_extraction.py. Submit from the project root or set REPO_ROOT." >&2
    return 2
  }
}

load_frontier_gpu_runtime() {
  module reset
  if [[ "$USE_OLCF_GPU_BIND" == "1" ]]; then
    module load olcf-container-tools
    module load apptainer-enable-gpu
  fi
}

activate_client_env() {
  module load miniforge3/23.11.0-0
  # Loading the module initializes conda on Frontier; this fallback keeps the
  # script robust if the shell function is not immediately available.
  if ! type conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  fi
  conda activate "$CLIENT_ENV"
  export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
}

visible_gpu_list() {
  seq -s, 0 $((TP_SIZE - 1))
}
