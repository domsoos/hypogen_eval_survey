from __future__ import annotations

from schema import ExtractionRecord, EXTRACTION_FIELDS
from utils import flatten_dict, normalize_text


def agreement_report(a: ExtractionRecord, b: ExtractionRecord) -> dict:
    fa = flatten_dict(a.model_dump())
    fb = flatten_dict(b.model_dump())
    by_field = {}
    agreed = 0
    for field in EXTRACTION_FIELDS:
        av = fa.get(field, "")
        bv = fb.get(field, "")
        same = normalize_text(av) == normalize_text(bv)
        by_field[field] = {
            "agree": same,
            "a": av,
            "b": bv,
        }
        agreed += int(same)
    return {
        "n_fields": len(EXTRACTION_FIELDS),
        "n_agree": agreed,
        "agreement_rate": agreed / len(EXTRACTION_FIELDS) if EXTRACTION_FIELDS else 1.0,
        "by_field": by_field,
    }
