from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from schema import EXTRACTION_FIELDS
from paths import HUMAN_DIR, RESULTS_DIR
from utils import normalize_text, token_f1, write_csv

STRUCTURED_FIELDS = {
    "screening.generates_research_hypotheses_or_ideas",
    "screening.reports_evaluation_of_generated_outputs",
    "screening.decision",
    "study_information.year_of_publication",
    "primary_evaluation.primary_metric_identification",
    "primary_evaluation.overall_result",
}
SET_FIELDS = {"evaluation_approach.evaluation_technique"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def norm_set(s):
    return {normalize_text(x) for x in str(s or "").split(";") if normalize_text(x)}


def main():
    ap = argparse.ArgumentParser(description="Compare Glimmer extractions against human-gold extractions.")
    ap.add_argument("--model", default=str(RESULTS_DIR / "model_extractions.csv"))
    ap.add_argument("--human", default=str(HUMAN_DIR / "human_reference.csv"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "comparison_summary.csv"))
    ap.add_argument("--details", default=str(RESULTS_DIR / "comparison_by_field.csv"))
    ap.add_argument("--by-paper", default=str(RESULTS_DIR / "comparison_by_paper.csv"))
    args = ap.parse_args()

    model_rows = read_csv(args.model)
    human_rows = read_csv(args.human)
    human = {r["paper_id"]: r for r in human_rows}

    details = []
    per_system = defaultdict(list)
    for m in model_rows:
        pid, system = m["paper_id"], m["system"]
        if pid not in human:
            continue
        h = human[pid]
        for field in EXTRACTION_FIELDS:
            hv, mv = h.get(field, ""), m.get(field, "")
            # Skip fields not yet annotated in human gold.
            if normalize_text(hv) == "":
                continue
            if field in SET_FIELDS:
                hs, ms = norm_set(hv), norm_set(mv)
                exact = hs == ms
                score = 1.0 if exact else (len(hs & ms) / len(hs | ms) if hs | ms else 1.0)
                metric = "set_jaccard"
            elif field in STRUCTURED_FIELDS:
                exact = normalize_text(hv) == normalize_text(mv)
                score = float(exact)
                metric = "exact"
            else:
                exact = normalize_text(hv) == normalize_text(mv)
                score = token_f1(hv, mv)
                metric = "token_f1_aux"
            row = {
                "paper_id": pid,
                "system": system,
                "field": field,
                "human": hv,
                "model": mv,
                "metric": metric,
                "score": round(score, 4),
                "exact": int(exact),
            }
            details.append(row)
            per_system[system].append(row)

    write_csv(args.details, details, ["paper_id", "system", "field", "human", "model", "metric", "score", "exact"])

    # Latency/token averages by system from model file.
    perf = defaultdict(lambda: {"lat": [], "tok": []})
    for m in model_rows:
        try:
            perf[m["system"]]["lat"].append(float(m.get("latency_sec") or 0))
        except ValueError:
            pass
        try:
            perf[m["system"]]["tok"].append(float(m.get("total_tokens") or 0))
        except ValueError:
            pass

    summary = []
    for system in sorted(per_system):
        rows = per_system[system]
        structured = [r["score"] for r in rows if r["metric"] in {"exact", "set_jaccard"}]
        text = [r["score"] for r in rows if r["metric"] == "token_f1_aux"]
        decisions = [r["score"] for r in rows if r["field"] == "screening.decision"]
        not_reported_rows = [
            r for r in rows
            if normalize_text(r["human"]) in {"not reported", "none reported", "none", "not applicable", "no applicable metric"}
        ]
        summary.append({
            "system": system,
            "n_scored_fields": len(rows),
            "structured_field_agreement": round(mean(structured), 4) if structured else "",
            "text_token_f1_aux": round(mean(text), 4) if text else "",
            "inclusion_exclusion_accuracy": round(mean(decisions), 4) if decisions else "",
            "not_reported_accuracy": round(mean(r["score"] for r in not_reported_rows), 4) if not_reported_rows else "",
            "avg_latency_sec": round(mean(perf[system]["lat"]), 3) if perf[system]["lat"] else "",
            "avg_total_tokens": round(mean(perf[system]["tok"]), 1) if perf[system]["tok"] else "",
        })

    fields = [
        "system", "n_scored_fields", "structured_field_agreement", "text_token_f1_aux",
        "inclusion_exclusion_accuracy", "not_reported_accuracy", "avg_latency_sec", "avg_total_tokens"
    ]
    write_csv(args.out, summary, fields)

    # Per-paper system comparison for diagnosing which papers are hard.
    per_paper = defaultdict(list)
    for r in details:
        per_paper[(r["paper_id"], r["system"])].append(r)
    by_paper_rows = []
    for (pid, system), rows in sorted(per_paper.items()):
        structured = [r["score"] for r in rows if r["metric"] in {"exact", "set_jaccard"}]
        text = [r["score"] for r in rows if r["metric"] == "token_f1_aux"]
        by_paper_rows.append({
            "paper_id": pid,
            "system": system,
            "n_scored_fields": len(rows),
            "structured_field_agreement": round(mean(structured), 4) if structured else "",
            "text_token_f1_aux": round(mean(text), 4) if text else "",
        })
    write_csv(args.by_paper, by_paper_rows, [
        "paper_id", "system", "n_scored_fields", "structured_field_agreement", "text_token_f1_aux"
    ])

    print(f"Wrote {args.out}")
    print(f"Wrote {args.details}")
    print(f"Wrote {args.by_paper}")
    print("Note: token-F1 for free-text fields is an auxiliary overlap measure, not a substitute for human semantic correctness review.")


if __name__ == "__main__":
    main()
