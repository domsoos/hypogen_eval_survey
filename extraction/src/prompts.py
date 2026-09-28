from __future__ import annotations

from schema import FORM_SCHEMA_TEXT, ExtractionRecord, EvidencePack, AuditResult, json_schema_for

COMMON_RULES = r"""
You are performing data extraction for a systematic review of scientific hypothesis/research-idea generation systems.
Be conservative, literal, evidence-grounded, and brief.

Critical rules:
1. Use only evidence from the supplied paper/evidence pack. Do not use outside knowledge.
2. Do NOT treat related-work systems as baselines unless they are actually compared in this study's experiments/results.
3. Distinguish a metric that is merely mentioned/defined from a metric that is actually computed/reported.
4. A statement that an evaluation method 'is used' does not prove that evaluation results are reported.
5. Human evaluation counts only when humans actually evaluate study outputs. Future-work proposals do not count.
6. Expert evaluation counts only when domain experts actually participate in evaluation.
7. LLM-as-a-judge counts only when an LLM judges/evaluates outputs, not when it only generates them.
8. Ground truth must be explicitly identifiable. Do not invent a gold standard from the setup.
9. Evaluation limitations must be explicitly reported by the authors. Do not add your own methodological critique.
10. Numerical values must match the paper exactly, including units/conditions.
11. If proposed and baseline values are not on the same metric/condition, do not present them as a direct comparison.
12. If no single primary metric clearly dominates, use 'Multiple metrics' rather than inventing one.
13. 'Not reported', 'None', 'No applicable metric', and 'Unclear' are valid and preferable to unsupported inference.
14. Research ideas/questions/opportunities can qualify as hypothesis-generation outputs if they are genuinely generated as candidate scientific directions; describe them using the paper's terminology.
15. For screening: a paper that generates ideas but does not actually report evaluation results of those generated outputs should be marked reports_evaluation_of_generated_outputs=false. If the review requires evaluated outputs, decision='exclude'.
16. Keep extraction prose concise but detailed enough to interpret without reopening the paper.
17. Evidence quotes should be short, exact snippets sufficient to verify the field. Record PDF page number (1-indexed) whenever possible.
18. Output valid JSON only. No markdown fences or commentary.
""".strip()


def single_extractor_prompt(paper_text: str) -> tuple[str, str]:
    system = COMMON_RULES + "\n\nYou are the single-agent extraction baseline. Read the full paper and fill the entire extraction schema."
    user = f"""
EXTRACTION FORM:\n{FORM_SCHEMA_TEXT}

OUTPUT JSON SCHEMA:\n{json_schema_for(ExtractionRecord)}

PAPER (page markers are authoritative):\n{paper_text}

Return one complete ExtractionRecord. Include field_evidence for every materially populated field and for important 'not reported' decisions when evidence/absence is discussed.
"""
    return system, user


def evidence_locator_prompt(paper_text: str) -> tuple[str, str]:
    system = COMMON_RULES + "\n\nYou are the Evidence Locator. Do not perform polished extraction. Find the strongest evidence relevant to every extraction field and screening decision."
    user = f"""
EXTRACTION FORM:\n{FORM_SCHEMA_TEXT}

OUTPUT JSON SCHEMA:\n{json_schema_for(EvidencePack)}

PAPER:\n{paper_text}

Build an EvidencePack. Search especially for: generated outputs, actual evaluation of those outputs, baselines used in experiments, evaluation participants, metrics, numerical results, ground truth, primary/main result, limitations, and traceability. Use support='not_found' when a key item is absent after inspecting the paper.
"""
    return system, user


def extractor_from_evidence_prompt(evidence_json: str, variant: str) -> tuple[str, str]:
    persona = {
        "A": "You are Extractor A. Prefer literal, directly supported wording and minimal inference.",
        "B": "You are Extractor B. Be especially skeptical about baselines, evaluation types, ground truth, and primary-metric claims.",
    }[variant]
    system = COMMON_RULES + "\n\n" + persona + " You must use only the EvidencePack below; do not assume facts that are not in it."
    user = f"""
EXTRACTION FORM:\n{FORM_SCHEMA_TEXT}

OUTPUT JSON SCHEMA:\n{json_schema_for(ExtractionRecord)}

EVIDENCE PACK:\n{evidence_json}

Return one complete ExtractionRecord. If evidence is insufficient, use the appropriate 'Not reported'/'None'/'Unclear' value rather than guessing.
"""
    return system, user


def audit_prompt(paper_text: str, draft_json: str, second_draft_json: str | None = None, evidence_json: str | None = None) -> tuple[str, str]:
    system = COMMON_RULES + r"""

You are the Evaluation Auditor. Your job is NOT to make the paper look better or the extraction more complete. Your job is to remove unsupported claims and correct extraction errors.
Audit every field against the paper. Pay special attention to:
- 'mentioned' versus actually evaluated/reported;
- related work versus experimental baselines;
- human/expert/LLM evaluation classification;
- exact metric/result values;
- whether the primary metric is truly dominant;
- author-reported limitations versus your own critique;
- whether generated ideas/hypotheses themselves were evaluated.
Return a corrected final extraction, plus a concise issue list.
"""
    extra = ""
    if second_draft_json:
        extra += f"\nSECOND INDEPENDENT DRAFT:\n{second_draft_json}\n"
    if evidence_json:
        extra += f"\nEVIDENCE PACK:\n{evidence_json}\n"
    user = f"""
OUTPUT JSON SCHEMA:\n{json_schema_for(AuditResult)}

DRAFT EXTRACTION:\n{draft_json}
{extra}
FULL PAPER:\n{paper_text}

Return an AuditResult. corrected_extraction must be complete and self-contained.
"""
    return system, user


def adjudicator_prompt(paper_text: str, evidence_json: str, draft_a_json: str, draft_b_json: str, audit_json: str) -> tuple[str, str]:
    system = COMMON_RULES + r"""

You are the final Adjudicator. Resolve disagreements between two independent extractions using the EvidencePack, Auditor findings, and the paper itself. Evidence outranks consensus. When uncertain, extract less rather than infer more. The final record must be internally consistent and concise.
"""
    user = f"""
OUTPUT JSON SCHEMA:\n{json_schema_for(ExtractionRecord)}

EVIDENCE PACK:\n{evidence_json}

EXTRACTOR A:\n{draft_a_json}

EXTRACTOR B:\n{draft_b_json}

AUDIT:\n{audit_json}

FULL PAPER:\n{paper_text}

Return only the final ExtractionRecord JSON.
"""
    return system, user
