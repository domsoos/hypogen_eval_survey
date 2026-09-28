from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from utils import normalize_text, write_csv
from paths import RESULTS_DIR

SYSTEM_LABELS = {
    "single": "S0 — Single agent",
    "single_audit": "S1 — Single + audit",
    "multi": "M1 — Full multi-agent",
}


def read_csv(path: str | Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Create the final S0/S1/M1 benchmark table.")
    ap.add_argument("--comparison", default=str(RESULTS_DIR / "comparison_summary.csv"))
    ap.add_argument("--model", default=str(RESULTS_DIR / "model_extractions.csv"))
    ap.add_argument("--manual", default=str(RESULTS_DIR / "manual_accuracy_summary.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "benchmark_table.csv"))
    ap.add_argument("--markdown", default=str(RESULTS_DIR / "benchmark_table.md"))
    args = ap.parse_args()

    comp = {r["system"]: r for r in read_csv(args.comparison)}
    model_rows = read_csv(args.model)

    # Manual correctness is optional until a human has adjudicated the side-by-side review.
    manual = {}
    if Path(args.manual).exists():
        manual = {r["system"]: r for r in read_csv(args.manual)}

    # Derive per-system runtime/token averages directly as a safeguard.
    perf = defaultdict(lambda: {"lat": [], "tok": []})
    for r in model_rows:
        system = r.get("system", "")
        lat = fnum(r.get("latency_sec"))
        tok = fnum(r.get("total_tokens"))
        if lat is not None:
            perf[system]["lat"].append(lat)
        if tok is not None:
            perf[system]["tok"].append(tok)

    rows = []
    for system in ["single", "single_audit", "multi"]:
        c = comp.get(system, {})
        m = manual.get(system, {})
        rows.append({
            "system": SYSTEM_LABELS[system],
            "human_field_accuracy": m.get("human_field_accuracy", ""),
            "structured_agreement": c.get("structured_field_agreement", ""),
            "free_text_token_f1_aux": c.get("text_token_f1_aux", ""),
            "unsupported_claim_rate": m.get("unsupported_claim_rate", ""),
            "not_reported_accuracy": c.get("not_reported_accuracy", ""),
            "avg_time_per_paper_sec": round(mean(perf[system]["lat"]), 3) if perf[system]["lat"] else "",
            "avg_tokens_per_paper": round(mean(perf[system]["tok"]), 1) if perf[system]["tok"] else "",
        })

    fields = list(rows[0].keys())
    write_csv(args.out, rows, fields)

    def fmt(v):
        return str(v) if str(v) else "—"

    md = [
        "| System | Human field accuracy | Structured agreement | Free-text token F1* | Unsupported claim rate | Not-reported accuracy | Avg time/paper (s) | Avg tokens/paper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            "| " + " | ".join([
                fmt(r["system"]), fmt(r["human_field_accuracy"]), fmt(r["structured_agreement"]),
                fmt(r["free_text_token_f1_aux"]), fmt(r["unsupported_claim_rate"]),
                fmt(r["not_reported_accuracy"]), fmt(r["avg_time_per_paper_sec"]),
                fmt(r["avg_tokens_per_paper"]),
            ]) + " |"
        )
    md.append("\n*Token F1 is an auxiliary lexical-overlap measure, not semantic correctness.")
    Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
