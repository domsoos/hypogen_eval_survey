from __future__ import annotations

import argparse
from pathlib import Path

from llm import GlimmerClient
from paths import PAPERS_DIR, RESULTS_DIR
from pipeline import run_one
from schema import EXTRACTION_FIELDS
from utils import flatten_dict, load_json, write_csv

MODES = ["single", "single_audit", "multi"]


def payload_to_row(payload: dict) -> dict:
    flat = flatten_dict(payload["extraction"])
    row = {
        "paper_id": payload["paper_id"],
        "system": payload["mode"],
        "decision": flat.get("screening.decision", ""),
        "latency_sec": payload["run_meta"].get("latency_sec", ""),
        "total_tokens": payload["run_meta"].get("total_tokens", ""),
    }
    for field in EXTRACTION_FIELDS:
        row[field] = flat.get(field, "")
    return row


def write_model_csv(out_dir: Path, rows: list[dict]) -> None:
    fields = ["paper_id", "system", "decision", "latency_sec", "total_tokens"] + EXTRACTION_FIELDS
    write_csv(out_dir / "model_extractions.csv", rows, fields)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Muse-Glimmer-30B systematic-review extraction benchmark.")
    ap.add_argument("--papers", default=str(PAPERS_DIR), help="Directory containing PDFs")
    ap.add_argument("--out", default=str(RESULTS_DIR), help="Results directory")
    ap.add_argument("--mode", choices=MODES + ["all"], default="all")
    ap.add_argument("--paper-ids", nargs="*", default=None, help="Optional PDF stems to run, e.g. 1244 347")
    ap.add_argument("--resume", action="store_true", help="Reuse existing per-paper/mode JSON files")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    args = ap.parse_args()

    paper_dir = Path(args.papers)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(paper_dir.glob("*.pdf"))
    if args.paper_ids:
        wanted = set(args.paper_ids)
        pdfs = [p for p in pdfs if p.stem in wanted]
        missing = sorted(wanted - {p.stem for p in pdfs})
        if missing:
            raise SystemExit(f"Requested paper IDs not found in {paper_dir}: {', '.join(missing)}")
    if not pdfs:
        raise SystemExit(f"No PDFs found in {paper_dir}")

    client = GlimmerClient(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    modes = MODES if args.mode == "all" else [args.mode]

    rows: list[dict] = []
    for pdf in pdfs:
        for mode in modes:
            result_path = out_dir / pdf.stem / f"{mode}.json"
            if args.resume and result_path.exists():
                print(f"[{pdf.name}] {mode}: reusing {result_path}", flush=True)
                payload = load_json(result_path)
            else:
                print(f"[{pdf.name}] running {mode}...", flush=True)
                payload = run_one(pdf, mode, out_dir, client)
            rows.append(payload_to_row(payload))
            # Keep a usable partial CSV even if a later model call fails.
            write_model_csv(out_dir, rows)

    print(f"Wrote {out_dir / 'model_extractions.csv'}")


if __name__ == "__main__":
    main()
