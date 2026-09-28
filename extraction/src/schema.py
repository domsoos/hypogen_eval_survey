from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

EvaluationTechnique = Literal[
    "Automatic/computational metrics",
    "Human evaluation",
    "Expert evaluation",
    "LLM-as-a-judge",
    "Task-based evaluation",
]

PrimaryMetricIdentification = Literal[
    "Explicitly stated",
    "Emphasized in main result",
    "Used to support main conclusion",
    "Multiple metrics",
    "Unable to determine",
    "No applicable metric",
]

OverallResult = Literal[
    "Better / positive",
    "Similar / no meaningful difference",
    "Worse / negative",
    "Mixed",
    "No baseline",
    "Unclear",
]

SupportLevel = Literal["direct", "indirect", "not_found", "contradictory"]
Decision = Literal["include", "exclude", "unclear"]


class Screening(BaseModel):
    generates_research_hypotheses_or_ideas: bool
    reports_evaluation_of_generated_outputs: bool
    decision: Decision
    reason: str


class StudyInformation(BaseModel):
    year_of_publication: str
    application_domain: str
    hypothesis_generation_method: str
    baseline: str


class EvaluationApproach(BaseModel):
    evaluation_technique: List[EvaluationTechnique] = Field(default_factory=list)
    ground_truth: str
    evaluation_procedure: str


class EvaluationMetrics(BaseModel):
    all_evaluation_metrics_reported: str


class PrimaryEvaluation(BaseModel):
    primary_metric: str
    primary_metric_identification: PrimaryMetricIdentification
    primary_metric_result: str
    baseline_result: str
    overall_result: OverallResult


class Interpretation(BaseModel):
    main_evaluation_finding: str
    evaluation_limitations: str


class Traceability(BaseModel):
    evidence_location: str


class EvidenceItem(BaseModel):
    field: str
    page: Optional[int] = None
    section: str = ""
    quote: str = ""
    support: SupportLevel


class ExtractionRecord(BaseModel):
    screening: Screening
    study_information: StudyInformation
    evaluation_approach: EvaluationApproach
    evaluation_metrics: EvaluationMetrics
    primary_evaluation: PrimaryEvaluation
    interpretation: Interpretation
    traceability: Traceability
    field_evidence: List[EvidenceItem] = Field(default_factory=list)


class EvidencePack(BaseModel):
    screening: Screening
    evidence: List[EvidenceItem]
    notes: List[str] = Field(default_factory=list)


class AuditIssue(BaseModel):
    field: str
    severity: Literal["error", "warning"]
    issue: str
    recommended_value: Optional[str] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)


class AuditResult(BaseModel):
    valid: bool
    issues: List[AuditIssue] = Field(default_factory=list)
    corrected_extraction: ExtractionRecord


EXTRACTION_FIELDS = [
    "screening.generates_research_hypotheses_or_ideas",
    "screening.reports_evaluation_of_generated_outputs",
    "screening.decision",
    "screening.reason",
    "study_information.year_of_publication",
    "study_information.application_domain",
    "study_information.hypothesis_generation_method",
    "study_information.baseline",
    "evaluation_approach.evaluation_technique",
    "evaluation_approach.ground_truth",
    "evaluation_approach.evaluation_procedure",
    "evaluation_metrics.all_evaluation_metrics_reported",
    "primary_evaluation.primary_metric",
    "primary_evaluation.primary_metric_identification",
    "primary_evaluation.primary_metric_result",
    "primary_evaluation.baseline_result",
    "primary_evaluation.overall_result",
    "interpretation.main_evaluation_finding",
    "interpretation.evaluation_limitations",
    "traceability.evidence_location",
]

FORM_SCHEMA_TEXT = r"""
# Study information
Year of publication
Application/domain
Hypothesis generation method
Baseline: method(s), model(s), system(s), or baseline(s) against which the proposed approach was compared. Enter 'None' if no baseline was used.

# Evaluation approach
Evaluation technique (select all):
1. Automatic/computational metrics
2. Human evaluation
3. Expert evaluation
4. LLM-as-a-judge
5. Task-based evaluation
Ground truth: what generated hypotheses were evaluated against, if anything.
Evaluation procedure: short description of how evaluation was performed.

# Evaluation metrics
All evaluation metrics reported: every evaluation metric or criterion reported in the paper, including reported values/results when available; separate with semicolons.

# Primary Evaluation
Primary metric: metric/criterion most strongly used to judge performance or success.
How primary metric was identified (select one):
1. Explicitly stated
2. Emphasized in main result
3. Used to support main conclusion
4. Multiple metrics
5. Unable to determine
6. No applicable metric
Primary metric result
Baseline result
Overall result (select one):
1. Better / positive
2. Similar / no meaningful difference
3. Worse / negative
4. Mixed
5. No baseline
6. Unclear

# Interpretation
Main evaluation finding/insight
Evaluation limitations: ONLY limitations explicitly reported by the authors. Enter 'None reported' if not stated.

# Traceability
Evidence location: page, section, table, figure, or appendix where the main evaluation evidence was found.
""".strip()


def json_schema_for(model: type[BaseModel]) -> Dict:
    return model.model_json_schema()
