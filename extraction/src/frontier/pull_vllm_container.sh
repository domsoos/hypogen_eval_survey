#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
source "$REPO_ROOT/src/frontier/common.sh"

mkdir -p "$CONTAINER_DIR"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$CACHE_DIR/apptainer}"
mkdir -p "$APPTAINER_CACHEDIR"

if [[ -f "$VLLM_SIF" ]]; then
  echo "Container already exists: $VLLM_SIF"
  exit 0
fi

# Do not load apptainer-enable-gpu while pulling/building containers; OLCF
# documents those helper modules for runtime use.
module reset

echo "Pulling vLLM ROCm v${VLLM_VERSION} -> $VLLM_SIF"
apptainer pull "$VLLM_SIF" "docker://vllm/vllm-openai-rocm:v${VLLM_VERSION}"
echo "Done: $VLLM_SIF"
