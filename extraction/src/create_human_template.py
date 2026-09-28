from __future__ import annotations

import argparse
from pathlib import Path

from schema import EXTRACTION_FIELDS
from paths import PAPERS_DIR, HUMAN_DIR
from utils import write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(PAPERS_DIR))
    ap.add_argument("--out", default=str(HUMAN_DIR / "human_gold.csv"))
    args = ap.parse_args()
    ids = [p.stem for p in sorted(Path(args.papers).glob("*.pdf"))]
    if not ids:
        # Default study set from the current review batch.
        ids = ["1244", "347", "393", "459", "605"]
    fields = ["paper_id", "annotator", "notes"] + EXTRACTION_FIELDS
    rows = [{"paper_id": pid, "annotator": "", "notes": ""} for pid in ids]
    write_csv(args.out, rows, fields)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
