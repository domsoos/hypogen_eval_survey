# Muse-Glimmer systematic-review extraction benchmark

This `src/` directory implements three extraction configurations using the same Muse-Glimmer-30B backend:

- **S0 — Single agent:** full paper + fixed extraction schema -> one extraction.
- **S1 — Single + audit:** S0 draft -> independent evidence-focused audit -> corrected extraction.
- **M1 — Full multi-agent:** evidence locator -> two independent extractors -> audit -> adjudication.

The experiment holds the model, PDFs, schema, decoding temperature, and human reference constant so the main variable is the agentic scaffold.

## Expected repository layout

```text
project_root/
├── human/
│   └── reviewer_1.csv
├── papers/
│   ├── 1244.pdf
│   ├── 347.pdf
│   ├── 393.pdf
│   ├── 459.pdf
│   └── 605.pdf
├── src/
│   ├── *.py
│   └── frontier/
└── results/          # generated
```

Python CLIs infer `project_root` from the location of `src/`, so they do not depend on the current working directory. Frontier scripts assume they are submitted from `project_root` unless `REPO_ROOT` is explicitly set.

## Local/unit checks

```bash
python -m pytest -q src/tests
python src/prepare_human_reference.py
```

The second command creates `human/human_reference.csv` from the completed Covidence export.

## Frontier

See `src/frontier/README_FRONTIER.md` for the exact setup and submission sequence.

## Main outputs

```text
results/
├── model_extractions.csv
├── comparison_by_field.csv
├── comparison_by_paper.csv
├── comparison_summary.csv
├── manual_review.csv
├── benchmark_table.csv
├── benchmark_table.md
├── run_metadata.txt
└── <paper_id>/
    ├── single.json
    ├── single_audit.json
    ├── single_audit_intermediate.json
    ├── multi.json
    └── multi_intermediate.json
```

`manual_review.csv` is intentionally separate from the completed human extraction. The existing reviewer CSV is the independent reference; the manual review sheet is only for later semantic scoring of each model field as `correct`, `partial`, `incorrect`, `unsupported`, or `missing`.
