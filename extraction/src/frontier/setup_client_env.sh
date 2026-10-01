#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
source "$REPO_ROOT/src/frontier/common.sh"
check_repo_layout

module reset
module load miniforge3/23.11.0-0

if [[ ! -d "$CLIENT_ENV" ]]; then
  conda create -y -p "$CLIENT_ENV" python=3.12 pip
fi
conda activate "$CLIENT_ENV"
python -m pip install --upgrade pip
python -m pip install -r "$SRC_DIR/requirements.txt"

python - <<'PY'
import fitz, openai, pydantic
print("Client environment OK")
print("PyMuPDF:", fitz.__doc__.splitlines()[0])
print("openai:", openai.__version__)
print("pydantic:", pydantic.__version__)
PY

echo "Client environment: $CLIENT_ENV"
