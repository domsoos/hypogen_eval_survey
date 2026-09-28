from __future__ import annotations

import argparse
import csv
from pathlib import Path

from schema import EXTRACTION_FIELDS
from paths import HUMAN_DIR
from utils import write_csv

# Covidence export -> internal extraction schema.
# Screening fields are intentionally left blank because the extraction export
# supplied by the reviewer does not contain screening decisions.
COLUMN_MAP = {
    "Year of publication": "study_information.year_of_publication",
    "Application/domain": "study_information.application_domain",
    "Hypothesis generation method": "study_information.hypothesis_generation_method",
    "Baseline": "study_information.baseline",
    "Evaluation technique": "evaluation_approach.evaluation_technique",
    "Ground truth": "evaluation_approach.ground_truth",
    "Evaluation procedure": "evaluation_approach.evaluation_procedure",
    "All evaluation metrics reported": "evaluation_metrics.all_evaluation_metrics_reported",
    "Primary metric": "primary_evaluation.primary_metric",
    "How was the primary metric identified?": "primary_evaluation.primary_metric_identification",
    "Primary metric result": "primary_evaluation.primary_metric_result",
    "Baseline result": "primary_evaluation.baseline_result",
    "Overall result": "primary_evaluation.overall_result",
    "Main evaluation finding/insight": "interpretation.main_evaluation_finding",
    "Evaluation limitations": "interpretation.evaluation_limitations",
    "Evidence location": "traceability.evidence_location",
}


def clean_header(s: str) -> str:
    return " ".join((s or "").strip().split())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert the completed Covidence reviewer CSV into the benchmark's human-reference format."
    )
    ap.add_argument("--input", default=str(HUMAN_DIR / "reviewer_1.csv"))
    ap.add_argument("--output", default=str(HUMAN_DIR / "human_reference.csv"))
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Human reviewer file not found: {in_path}")

    with in_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)
        raw_headers = reader.fieldnames or []

    header_lookup = {clean_header(h): h for h in raw_headers}
    missing = [src for src in COLUMN_MAP if clean_header(src) not in header_lookup]
    if missing:
        raise SystemExit(
            "Missing expected Covidence columns: " + ", ".join(missing) +
            "\nObserved headers: " + ", ".join(raw_headers)
        )

    rows = []
    for raw in raw_rows:
        cov_col = header_lookup.get("Covidence #")
        if not cov_col:
            raise SystemExit("Missing 'Covidence #' column; cannot match reviewer rows to PDF filenames.")
        paper_id = (raw.get(cov_col) or "").strip()
        if not paper_id:
            continue

        row = {
            "paper_id": paper_id,
            "study_id": (raw.get(header_lookup.get("Study ID", "")) or "").strip(),
            "title": (raw.get(header_lookup.get("Title", "")) or "").strip(),
            "reviewer_name": (raw.get(header_lookup.get("Reviewer Name", "")) or "").strip(),
        }
        for field in EXTRACTION_FIELDS:
            row[field] = ""
        for source, target in COLUMN_MAP.items():
            source_actual = header_lookup[clean_header(source)]
            row[target] = (raw.get(source_actual) or "").strip()
        rows.append(row)

    if not rows:
        raise SystemExit("No reviewer rows found.")

    fields = ["paper_id", "study_id", "title", "reviewer_name"] + EXTRACTION_FIELDS
    write_csv(args.output, rows, fields)
    print(f"Wrote {args.output} ({len(rows)} papers)")
    print("Screening fields were left blank because they are not present in the Covidence extraction export.")


if __name__ == "__main__":
    main()
