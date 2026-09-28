from __future__ import annotations

from pathlib import Path

# This repository uses:
#   project_root/
#     human/
#     papers/
#     results/        (generated)
#     logs/           (generated)
#     src/
#
# Resolve from this file so the Python CLIs work regardless of current directory.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
HUMAN_DIR = PROJECT_ROOT / "human"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
