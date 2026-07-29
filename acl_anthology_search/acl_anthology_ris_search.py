#!/usr/bin/env python3
"""Search ACL Anthology metadata and export one RIS file per query.

The script uses the official ``acl-anthology`` Python package. It searches
paper titles and available abstracts, writes separate RIS files for each
selected query, and also writes a deduplicated combined RIS file and audit
files for reproducibility.

Examples
--------
Run the four scientific-hypothesis queries (recommended for the core corpus):

    python acl_anthology_ris_search.py --queries core

Run every query, including the broad methodological LLM-judge query:

    python acl_anthology_ris_search.py --queries all

Run selected queries only:

    python acl_anthology_ris_search.py --queries Q1 Q2 Q3
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class QuerySpec:
    """A Boolean query represented as AND-connected blocks of OR terms."""

    query_id: str
    slug: str
    label: str
    corpus: str
    blocks: tuple[tuple[str, ...], ...]


MODEL_TERMS = (
    "large language model*",
    "LLM",
    "LLMs",
    "foundation model*",
    "generative artificial intelligence",
    "generative AI",
    "ChatGPT",
    "GPT-3",
    "GPT-3.5",
    "GPT-4",
    "GPT-4o",
    "Claude",
    "Gemini",
    "Llama",
)

SCIENTIFIC_OBJECT_TERMS = (
    "scientific hypothesis",
    "scientific hypotheses",
    "research hypothesis",
    "research hypotheses",
    "hypothesis generation",
    "hypothesis discovery",
    "hypothesis composition",
    "hypothesis ranking",
    "scientific idea",
    "scientific ideas",
    "scientific idea generation",
    "scientific ideation",
    "research idea",
    "research ideas",
    "research idea generation",
    "research ideation",
    "research question generation",
    "research proposal",
    "research proposals",
)

EVALUATION_TERMS = (
    "evaluat*",
    "assess*",
    "judg*",
    "scor*",
    "rank*",
    "select*",
    "filter*",
    "critic*",
    "compar*",
    "benchmark*",
    "metric*",
    "rubric*",
    "validat*",
    "reliab*",
    "agreement",
    "calibration",
    "novelty",
    "originality",
    "feasibility",
    "plausibility",
    "testability",
    "truthfulness",
    "correctness",
    "groundedness",
    "relevance",
    "usefulness",
    "significance",
    "scientific impact",
    "human evaluation",
    "expert evaluation",
    "human-LLM agreement",
    "human-AI collaboration",
    "human-in-the-loop",
)

JUDGE_TERMS = (
    "LLM-as-a-judge",
    "LLM as a judge",
    "LLM judge",
    "LLM judges",
    "language model judge",
    "language model judges",
    "LLM evaluator",
    "LLM evaluators",
    "language model evaluator",
    "language model evaluators",
    "AI evaluator",
    "AI evaluators",
    "model-based evaluation",
    "model-based evaluator",
    "automated evaluation",
    "automated evaluator",
    "automated scientific reviewer",
)

JUDGE_SCIENCE_TERMS = (
    "scientific hypothesis",
    "scientific hypotheses",
    "research hypothesis",
    "research hypotheses",
    "hypothesis generation",
    "hypothesis ranking",
    "scientific idea",
    "scientific ideas",
    "research idea",
    "research ideas",
    "research question",
    "research questions",
    "research proposal",
    "research proposals",
    "scientific discovery",
    "scientific research",
)

AGENT_TERMS = (
    "AI scientist",
    "AI scientists",
    "AI co-scientist",
    "AI co-scientists",
    "autonomous scientist",
    "autonomous scientists",
    "automated scientist",
    "automated scientists",
    "virtual scientist",
    "virtual scientists",
    "scientific agent",
    "scientific agents",
    "research agent",
    "research agents",
    "autonomous research agent",
    "scientific discovery agent",
    "multi-agent scientific system",
    "autonomous scientific discovery",
    "automated scientific discovery",
)

AGENT_OBJECT_TERMS = (
    "hypothesis",
    "hypotheses",
    "hypothesis generation",
    "hypothesis ranking",
    "ideation",
    "scientific idea",
    "scientific ideas",
    "research idea",
    "research ideas",
    "research proposal",
    "research proposals",
)

AGENT_EVALUATION_TERMS = (
    "evaluat*",
    "assess*",
    "judg*",
    "scor*",
    "rank*",
    "critic*",
    "review*",
    "select*",
    "benchmark*",
    "metric*",
    "rubric*",
    "validat*",
    "novelty",
    "feasibility",
    "plausibility",
    "human evaluation",
    "expert evaluation",
)

TEMPORAL_TERMS = (
    "temporal evaluation",
    "temporal validation",
    "time-aware evaluation",
    "time-aware validation",
    "retrospective evaluation",
    "retrospective validation",
    "prospective evaluation",
    "prospective validation",
    "chronological evaluation",
    "chronological split",
    "temporal split",
    "time-sliced evaluation",
    "time cutoff",
    "temporal cutoff",
    "knowledge cutoff",
    "training cutoff",
    "literature cutoff",
    "post-cutoff",
    "future publication",
    "future publications",
    "future knowledge",
    "future discovery",
    "historical novelty",
    "novelty at the time",
    "literature-grounded novelty",
    "literature-grounded evaluation",
    "evidence-grounded evaluation",
)

JUDGE_METHOD_TERMS = (
    "reliab*",
    "valid*",
    "agreement",
    "correlation",
    "calibration",
    "consistency",
    "robustness",
    "reproducibility",
    "bias",
    "biases",
    "position bias",
    "order bias",
    "verbosity bias",
    "length bias",
    "self-preference",
    "self-bias",
    "reference bias",
    "presentation bias",
    "human alignment",
    "human agreement",
    "expert agreement",
    "inter-rater agreement",
    "meta-evaluation",
    "meta evaluation",
)

QUERY_SPECS: Mapping[str, QuerySpec] = {
    "Q1": QuerySpec(
        query_id="Q1",
        slug="core_hypothesis_evaluation",
        label="Core scientific hypothesis generation and evaluation",
        corpus="core",
        blocks=(MODEL_TERMS, SCIENTIFIC_OBJECT_TERMS, EVALUATION_TERMS),
    ),
    "Q2": QuerySpec(
        query_id="Q2",
        slug="explicit_llm_judge",
        label="Explicit LLM-as-a-judge methods in scientific research",
        corpus="core",
        blocks=(JUDGE_TERMS, JUDGE_SCIENCE_TERMS),
    ),
    "Q3": QuerySpec(
        query_id="Q3",
        slug="scientific_agents",
        label="AI scientists and scientific agents",
        corpus="core",
        blocks=(MODEL_TERMS, AGENT_TERMS, AGENT_OBJECT_TERMS, AGENT_EVALUATION_TERMS),
    ),
    "Q4": QuerySpec(
        query_id="Q4",
        slug="temporal_literature_grounded",
        label="Temporal and literature-grounded evaluation",
        corpus="core-targeted",
        blocks=(MODEL_TERMS, SCIENTIFIC_OBJECT_TERMS, TEMPORAL_TERMS),
    ),
    "Q5": QuerySpec(
        query_id="Q5",
        slug="llm_judge_reliability_bias",
        label="General LLM-judge reliability and bias",
        corpus="methodological-context",
        blocks=(JUDGE_TERMS, JUDGE_METHOD_TERMS),
    ),
}


def normalize_text(value: Any) -> str:
    """Normalize text for case-insensitive phrase and stem matching."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    chars = [character if character.isalnum() else " " for character in text]
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def compile_term(term: str) -> re.Pattern[str]:
    """Compile one search term, supporting a trailing ``*`` wildcard."""

    stripped = term.strip().strip('"')
    has_wildcard = stripped.endswith("*")
    if has_wildcard:
        stripped = stripped[:-1]

    normalized = normalize_text(stripped)
    if not normalized:
        raise ValueError(f"Empty search term after normalization: {term!r}")

    tokens = normalized.split()
    token_patterns = [re.escape(token) for token in tokens]
    if has_wildcard:
        token_patterns[-1] += r"\w*"

    expression = r"(?<!\w)" + r"\s+".join(token_patterns) + r"(?!\w)"
    return re.compile(expression, flags=re.IGNORECASE)


COMPILED_QUERIES: Mapping[str, tuple[tuple[tuple[str, re.Pattern[str]], ...], ...]] = {
    query_id: tuple(
        tuple((term, compile_term(term)) for term in block) for block in spec.blocks
    )
    for query_id, spec in QUERY_SPECS.items()
}


def match_query(query_id: str, normalized_record_text: str) -> tuple[bool, list[list[str]]]:
    """Return whether a record matches and the terms matched in each block."""

    matched_by_block: list[list[str]] = []
    for block in COMPILED_QUERIES[query_id]:
        block_matches = [term for term, pattern in block if pattern.search(normalized_record_text)]
        if not block_matches:
            return False, []
        matched_by_block.append(block_matches)
    return True, matched_by_block


def clean_ris_value(value: Any) -> str:
    """Flatten a value to a single clean RIS line."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def author_to_ris(author_spec: Any) -> str:
    """Convert an ACL Anthology NameSpecification to ``Last, First``."""

    name = getattr(author_spec, "name", author_spec)
    first = clean_ris_value(getattr(name, "first", ""))
    last = clean_ris_value(getattr(name, "last", ""))
    if last and first:
        return f"{last}, {first}"
    return last or first or clean_ris_value(name)


def safe_attr(obj: Any, attribute: str, default: Any = "") -> Any:
    """Read a possibly optional property without stopping the search."""

    try:
        value = getattr(obj, attribute, default)
    except Exception:
        return default
    return default if value is None else value


def paper_title(paper: Any) -> str:
    return clean_ris_value(safe_attr(paper, "title"))


def paper_abstract(paper: Any) -> str:
    return clean_ris_value(safe_attr(paper, "abstract"))


def paper_url(paper: Any) -> str:
    url = clean_ris_value(safe_attr(paper, "web_url"))
    if url:
        return url
    full_id = clean_ris_value(safe_attr(paper, "full_id"))
    return f"https://aclanthology.org/{full_id}/" if full_id else ""


def paper_pdf_url(paper: Any) -> str:
    pdf = safe_attr(paper, "pdf", None)
    if pdf is not None:
        url = clean_ris_value(safe_attr(pdf, "url"))
        if url:
            return url
    full_id = clean_ris_value(safe_attr(paper, "full_id"))
    return f"https://aclanthology.org/{full_id}.pdf" if full_id else ""


def paper_volume_title(paper: Any) -> str:
    parent = safe_attr(paper, "parent", None)
    if parent is None:
        return ""
    return clean_ris_value(safe_attr(parent, "title"))


def paper_type(paper: Any) -> str:
    return "JOUR" if clean_ris_value(safe_attr(paper, "journal_title")) else "CONF"


def split_pages(pages: str) -> tuple[str, str]:
    cleaned = clean_ris_value(pages)
    if not cleaned:
        return "", ""
    parts = re.split(r"\s*[-–—]+\s*", cleaned, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (cleaned, "")


def ris_record(paper: Any, query_ids: Sequence[str]) -> str:
    """Create a Covidence-compatible RIS record."""

    lines: list[str] = [f"TY  - {paper_type(paper)}"]
    title = paper_title(paper)
    if title:
        lines.append(f"TI  - {title}")

    for author in safe_attr(paper, "authors", ()):
        author_value = author_to_ris(author)
        if author_value:
            lines.append(f"AU  - {author_value}")

    year = clean_ris_value(safe_attr(paper, "year"))
    if year:
        lines.append(f"PY  - {year}")

    month = clean_ris_value(safe_attr(paper, "month"))
    if month:
        lines.append(f"DA  - {year}/{month}" if year else f"DA  - {month}")

    journal_title = clean_ris_value(safe_attr(paper, "journal_title"))
    volume_title = paper_volume_title(paper)
    if journal_title:
        lines.append(f"JF  - {journal_title}")
    elif volume_title:
        lines.append(f"T2  - {volume_title}")

    journal_volume = clean_ris_value(safe_attr(paper, "journal_volume"))
    journal_issue = clean_ris_value(safe_attr(paper, "journal_issue"))
    if journal_volume:
        lines.append(f"VL  - {journal_volume}")
    if journal_issue:
        lines.append(f"IS  - {journal_issue}")

    start_page, end_page = split_pages(clean_ris_value(safe_attr(paper, "pages")))
    if start_page:
        lines.append(f"SP  - {start_page}")
    if end_page:
        lines.append(f"EP  - {end_page}")

    publisher = clean_ris_value(safe_attr(paper, "publisher"))
    if publisher:
        lines.append(f"PB  - {publisher}")

    doi = clean_ris_value(safe_attr(paper, "doi"))
    if doi:
        lines.append(f"DO  - {doi}")

    abstract = paper_abstract(paper)
    if abstract:
        lines.append(f"AB  - {abstract}")

    language = clean_ris_value(safe_attr(paper, "language_name"))
    if language:
        lines.append(f"LA  - {language}")

    full_id = clean_ris_value(safe_attr(paper, "full_id"))
    if full_id:
        lines.append(f"AN  - {full_id}")
        lines.append(f"ID  - {full_id}")

    url = paper_url(paper)
    if url:
        lines.append(f"UR  - {url}")

    pdf_url = paper_pdf_url(paper)
    if pdf_url:
        lines.append(f"L1  - {pdf_url}")

    for query_id in query_ids:
        spec = QUERY_SPECS[query_id]
        lines.append(f"KW  - ACL search {query_id}: {spec.label}")

    lines.append(
        "N1  - Retrieved from ACL Anthology metadata; matched query IDs: "
        + ", ".join(query_ids)
    )
    lines.append("ER  -")
    return "\n".join(lines) + "\n"


def year_in_range(paper: Any, min_year: int | None, max_year: int | None) -> bool:
    raw_year = clean_ris_value(safe_attr(paper, "year"))
    try:
        year = int(raw_year)
    except (TypeError, ValueError):
        return min_year is None and max_year is None
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def choose_queries(raw_queries: Sequence[str]) -> list[str]:
    """Expand ``core`` and ``all`` aliases and validate query IDs."""

    normalized = [item.upper() for item in raw_queries]
    if "ALL" in normalized:
        return list(QUERY_SPECS)
    if "CORE" in normalized:
        expanded = ["Q1", "Q2", "Q3", "Q4"]
        expanded.extend(item for item in normalized if item not in {"CORE", "ALL"})
        normalized = expanded

    unknown = sorted(set(normalized) - set(QUERY_SPECS))
    if unknown:
        valid = ", ".join([*QUERY_SPECS, "core", "all"])
        raise ValueError(f"Unknown query selection {unknown}. Valid values: {valid}")

    # Preserve canonical order and remove duplicates.
    selected = set(normalized)
    return [query_id for query_id in QUERY_SPECS if query_id in selected]


def load_anthology(repo_path: Path | None, verbose: bool) -> Any:
    try:
        from acl_anthology import Anthology
    except ImportError as exc:
        raise RuntimeError(
            "The acl-anthology package is not installed. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {"verbose": verbose}
    if repo_path is not None:
        kwargs["path"] = repo_path
    return Anthology.from_repo(**kwargs)


def git_snapshot(datadir: Any) -> dict[str, str]:
    """Record the repository root and commit when available."""

    result = {"repository_root": "", "git_commit": ""}
    try:
        data_path = Path(datadir).resolve()
        root = subprocess.run(
            ["git", "-C", str(data_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result = {"repository_root": root, "git_commit": commit}
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return result


def write_ris(path: Path, papers: Iterable[Any], query_map: Mapping[str, Sequence[str]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for paper in papers:
            full_id = clean_ris_value(safe_attr(paper, "full_id"))
            handle.write(ris_record(paper, query_map[full_id]))
            handle.write("\n")
            count += 1
    return count


def paper_sort_key(paper: Any) -> tuple[int, str, str]:
    raw_year = clean_ris_value(safe_attr(paper, "year"))
    try:
        year = int(raw_year)
    except ValueError:
        year = 0
    return (-year, paper_title(paper).casefold(), clean_ris_value(safe_attr(paper, "full_id")))


def run_search(args: argparse.Namespace) -> int:
    selected_queries = choose_queries(args.queries)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    anthology = load_anthology(
        Path(args.repo_path).expanduser().resolve() if args.repo_path else None,
        verbose=not args.quiet,
    )

    results: dict[str, dict[str, Any]] = {query_id: {} for query_id in selected_queries}
    matched_terms: dict[str, dict[str, list[list[str]]]] = defaultdict(dict)
    scanned = 0
    missing_abstracts = 0

    try:
        papers_iterator: Iterator[Any] = anthology.papers()
        for paper in papers_iterator:
            scanned += 1
            if safe_attr(paper, "is_deleted", False) or safe_attr(paper, "is_frontmatter", False):
                continue
            if not year_in_range(paper, args.min_year, args.max_year):
                continue

            title = paper_title(paper)
            abstract = "" if args.title_only else paper_abstract(paper)
            if not abstract:
                missing_abstracts += 1
            normalized_record_text = normalize_text(f"{title} {abstract}")
            if not normalized_record_text:
                continue

            full_id = clean_ris_value(safe_attr(paper, "full_id"))
            if not full_id:
                continue

            for query_id in selected_queries:
                matched, details = match_query(query_id, normalized_record_text)
                if matched:
                    results[query_id][full_id] = paper
                    matched_terms[full_id][query_id] = details
    except Exception as exc:
        raise RuntimeError(
            "ACL Anthology metadata iteration failed. First update the package with "
            "`python -m pip install --upgrade acl-anthology`, then rerun. "
            "If the problem persists, delete the local ACL repository cache or pass "
            "a fresh directory with --repo-path."
        ) from exc

    counts: dict[str, int] = {}
    all_papers: dict[str, Any] = {}
    all_query_ids: dict[str, list[str]] = defaultdict(list)

    for query_id in selected_queries:
        spec = QUERY_SPECS[query_id]
        papers = sorted(results[query_id].values(), key=paper_sort_key)
        query_map = {
            clean_ris_value(safe_attr(paper, "full_id")): [query_id] for paper in papers
        }
        path = output_dir / f"acl_{query_id.lower()}_{spec.slug}.ris"
        counts[query_id] = write_ris(path, papers, query_map)

        for paper in papers:
            full_id = clean_ris_value(safe_attr(paper, "full_id"))
            all_papers[full_id] = paper
            all_query_ids[full_id].append(query_id)

    combined_papers = sorted(all_papers.values(), key=paper_sort_key)
    combined_path = output_dir / "acl_all_selected_queries_deduplicated.ris"
    combined_count = write_ris(combined_path, combined_papers, all_query_ids)

    audit_path = output_dir / "acl_search_audit.csv"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "acl_id",
                "title",
                "year",
                "abstract_available",
                "doi",
                "url",
                "matched_queries",
                "matched_terms_by_block",
            ],
        )
        writer.writeheader()
        for paper in combined_papers:
            full_id = clean_ris_value(safe_attr(paper, "full_id"))
            details = {
                query_id: matched_terms[full_id][query_id]
                for query_id in all_query_ids[full_id]
            }
            writer.writerow(
                {
                    "acl_id": full_id,
                    "title": paper_title(paper),
                    "year": clean_ris_value(safe_attr(paper, "year")),
                    "abstract_available": bool(paper_abstract(paper)),
                    "doi": clean_ris_value(safe_attr(paper, "doi")),
                    "url": paper_url(paper),
                    "matched_queries": "; ".join(all_query_ids[full_id]),
                    "matched_terms_by_block": json.dumps(details, ensure_ascii=False),
                }
            )

    try:
        package_version = importlib.metadata.version("acl-anthology")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"

    snapshot = git_snapshot(safe_attr(anthology, "datadir", ""))
    manifest = {
        "source": "ACL Anthology",
        "search_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "search_fields": "title only" if args.title_only else "title and available abstract",
        "selected_queries": selected_queries,
        "query_definitions": {
            query_id: {
                "label": QUERY_SPECS[query_id].label,
                "corpus": QUERY_SPECS[query_id].corpus,
                "boolean_structure": " AND ".join(
                    "(" + " OR ".join(block) + ")"
                    for block in QUERY_SPECS[query_id].blocks
                ),
            }
            for query_id in selected_queries
        },
        "year_limits": {"minimum": args.min_year, "maximum": args.max_year},
        "records_scanned": scanned,
        "records_without_abstract_during_scan": missing_abstracts,
        "result_counts": counts,
        "combined_deduplicated_count": combined_count,
        "acl_anthology_package_version": package_version,
        "python_version": platform.python_version(),
        **snapshot,
    }
    manifest_path = output_dir / "acl_search_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Scanned {scanned:,} ACL Anthology records.")
    for query_id in selected_queries:
        print(f"{query_id}: {counts[query_id]:,} records -> acl_{query_id.lower()}_{QUERY_SPECS[query_id].slug}.ris")
    print(f"Combined deduplicated: {combined_count:,} records -> {combined_path.name}")
    print(f"Audit: {audit_path.name}")
    print(f"Manifest: {manifest_path.name}")
    print(f"Output directory: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search ACL Anthology titles/abstracts and export RIS files by query."
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["core"],
        help=(
            "Query IDs to run: Q1 Q2 Q3 Q4 Q5. Use 'core' for Q1-Q4 "
            "or 'all' for Q1-Q5. Default: core."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="acl_ris_results",
        help="Directory for RIS and audit files (default: acl_ris_results).",
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help=(
            "Optional persistent local path for the ACL Anthology Git repository. "
            "Using a fixed path improves reproducibility and avoids repeated downloads."
        ),
    )
    parser.add_argument("--min-year", type=int, default=None, help="Optional minimum publication year.")
    parser.add_argument("--max-year", type=int, default=None, help="Optional maximum publication year.")
    parser.add_argument(
        "--title-only",
        action="store_true",
        help="Search titles only instead of titles plus available abstracts.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress package progress bars.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_search(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
