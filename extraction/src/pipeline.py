from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from agents import agreement_report
from llm import GlimmerClient
from pdf_utils import format_paper_with_page_markers, paper_stats
from prompts import (
    single_extractor_prompt,
    evidence_locator_prompt,
    extractor_from_evidence_prompt,
    audit_prompt,
    adjudicator_prompt,
)
from schema import ExtractionRecord, EvidencePack, AuditResult
from utils import dump_json, ensure_dir

Mode = Literal["single", "single_audit", "multi"]


def _meta_sum(metas) -> dict:
    return {
        "calls": len(metas),
        "latency_sec": round(sum(m.latency_sec for m in metas), 4),
        "prompt_tokens": sum(m.prompt_tokens or 0 for m in metas),
        "completion_tokens": sum(m.completion_tokens or 0 for m in metas),
        "total_tokens": sum(m.total_tokens or 0 for m in metas),
    }


def run_single(pdf_path: Path, client: GlimmerClient):
    paper = format_paper_with_page_markers(pdf_path)
    s, u = single_extractor_prompt(paper)
    rec, meta, _ = client.call(s, u, ExtractionRecord, seed=11)
    return rec, _meta_sum([meta]), {}


def run_single_audit(pdf_path: Path, client: GlimmerClient):
    paper = format_paper_with_page_markers(pdf_path)
    s, u = single_extractor_prompt(paper)
    draft, m1, _ = client.call(s, u, ExtractionRecord, seed=11)
    s, u = audit_prompt(paper, draft.model_dump_json())
    audit, m2, _ = client.call(s, u, AuditResult, seed=17)
    return audit.corrected_extraction, _meta_sum([m1, m2]), {
        "draft": draft.model_dump(),
        "audit": audit.model_dump(),
    }


def run_multi(pdf_path: Path, client: GlimmerClient):
    paper = format_paper_with_page_markers(pdf_path)
    s, u = evidence_locator_prompt(paper)
    evidence, m1, _ = client.call(s, u, EvidencePack, seed=7)
    evidence_json = evidence.model_dump_json()

    def do_extract(variant: str, seed: int):
        es, eu = extractor_from_evidence_prompt(evidence_json, variant)
        return client.call(es, eu, ExtractionRecord, seed=seed)

    # Independent contexts. Concurrent calls can be disabled by setting GLIMMER_SERIAL=1.
    import os
    if os.getenv("GLIMMER_SERIAL", "0") == "1":
        a, m2, _ = do_extract("A", 11)
        b, m3, _ = do_extract("B", 29)
    else:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fa = ex.submit(do_extract, "A", 11)
            fb = ex.submit(do_extract, "B", 29)
            a, m2, _ = fa.result()
            b, m3, _ = fb.result()

    s, u = audit_prompt(
        paper,
        a.model_dump_json(),
        second_draft_json=b.model_dump_json(),
        evidence_json=evidence_json,
    )
    audit, m4, _ = client.call(s, u, AuditResult, seed=41)

    s, u = adjudicator_prompt(
        paper,
        evidence_json,
        a.model_dump_json(),
        b.model_dump_json(),
        audit.model_dump_json(),
    )
    final, m5, _ = client.call(s, u, ExtractionRecord, seed=53)

    agreement = agreement_report(a, b)
    return final, _meta_sum([m1, m2, m3, m4, m5]), {
        "evidence_pack": evidence.model_dump(),
        "extractor_a": a.model_dump(),
        "extractor_b": b.model_dump(),
        "agreement": agreement,
        "audit": audit.model_dump(),
    }


def run_one(pdf_path: str | Path, mode: Mode, out_dir: str | Path, client: GlimmerClient):
    pdf_path = Path(pdf_path)
    out_dir = ensure_dir(out_dir)
    paper_id = pdf_path.stem
    paper_dir = ensure_dir(out_dir / paper_id)

    if mode == "single":
        record, meta, intermediate = run_single(pdf_path, client)
    elif mode == "single_audit":
        record, meta, intermediate = run_single_audit(pdf_path, client)
    elif mode == "multi":
        record, meta, intermediate = run_multi(pdf_path, client)
    else:
        raise ValueError(mode)

    payload = {
        "paper_id": paper_id,
        "pdf": str(pdf_path),
        "mode": mode,
        "model": client.model,
        "paper_stats": paper_stats(pdf_path),
        "run_meta": meta,
        "extraction": record.model_dump(),
    }
    dump_json(paper_dir / f"{mode}.json", payload)
    if intermediate:
        dump_json(paper_dir / f"{mode}_intermediate.json", intermediate)
    return payload
