from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_covidence_header_mapping_accepts_trailing_space(tmp_path: Path):
    src_dir = Path(__file__).resolve().parents[1]
    inp = tmp_path / "reviewer.csv"
    out = tmp_path / "human_reference.csv"
    headers = [
        "Covidence #", "Study ID", "Title", "Reviewer Name", "Year of publication",
        "Application/domain", "Hypothesis generation method", "Baseline", "Evaluation technique",
        "Ground truth", "Evaluation procedure", "All evaluation metrics reported", "Primary metric",
        "How was the primary metric identified? ", "Primary metric result", "Baseline result",
        "Overall result", "Main evaluation finding/insight", "Evaluation limitations", "Evidence location",
    ]
    row = {h: "x" for h in headers}
    row.update({"Covidence #": "1244", "Year of publication": "2026", "Evaluation technique": "Automatic/computational metrics; human evaluation"})
    with inp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerow(row)

    subprocess.run(
        [sys.executable, str(src_dir / "prepare_human_reference.py"), "--input", str(inp), "--output", str(out)],
        check=True,
        cwd=src_dir,
    )
    with out.open(newline="", encoding="utf-8") as f:
        got = next(csv.DictReader(f))
    assert got["paper_id"] == "1244"
    assert got["study_information.year_of_publication"] == "2026"
    assert got["evaluation_approach.evaluation_technique"] == "Automatic/computational metrics; human evaluation"
