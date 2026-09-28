from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean
from utils import normalize_text, write_csv
from paths import RESULTS_DIR

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "unsupported": 0.0, "missing": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=str(RESULTS_DIR / "manual_review.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "manual_accuracy_summary.csv"))
    args = ap.parse_args()
    with open(args.review, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = defaultdict(list)
    unsupported = defaultdict(list)
    for r in rows:
        j = normalize_text(r.get("human_judgment", ""))
        if j not in SCORES:
            continue
        by[r["system"]].append(SCORES[j])
        unsupported[r["system"]].append(1.0 if j == "unsupported" else 0.0)
    out = []
    for system in sorted(by):
        out.append({
            "system": system,
            "human_field_accuracy": round(mean(by[system]), 4),
            "unsupported_claim_rate": round(mean(unsupported[system]), 4),
            "n_human_scored_fields": len(by[system]),
        })
    write_csv(args.out, out, ["system", "human_field_accuracy", "unsupported_claim_rate", "n_human_scored_fields"])
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
