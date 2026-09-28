from __future__ import annotations

import argparse
import csv
from pathlib import Path

from schema import EXTRACTION_FIELDS
from paths import HUMAN_DIR, RESULTS_DIR
from utils import normalize_text, write_csv


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="Create a side-by-side manual correctness review sheet.")
    ap.add_argument("--model", default=str(RESULTS_DIR / "model_extractions.csv"))
    ap.add_argument("--human", default=str(HUMAN_DIR / "human_reference.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "manual_review.csv"))
    args = ap.parse_args()

    model_rows = read_csv(args.model)
    human_rows = {r["paper_id"]: r for r in read_csv(args.human)}
    rows = []
    for m in model_rows:
        h = human_rows.get(m["paper_id"], {})
        for field in EXTRACTION_FIELDS:
            if normalize_text(h.get(field, "")) == "":
                continue
            rows.append({
                "paper_id": m["paper_id"],
                "system": m["system"],
                "field": field,
                "human_reference": h.get(field, ""),
                "model_value": m.get(field, ""),
                "human_judgment": "",  # correct | partial | incorrect | unsupported | missing
                "reviewer_notes": "",
            })
    write_csv(args.out, rows, ["paper_id", "system", "field", "human_reference", "model_value", "human_judgment", "reviewer_notes"])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
