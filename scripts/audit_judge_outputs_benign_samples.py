#!/usr/bin/env python3
"""Audit saved RIGJ ASR/AHS judgements against benchmark source metadata.

ASR mode preserves the original WildJailbreak benign-sample audit. AHS mode
emits the paper-five subset: AdvBench, official HarmBench standard,
JailbreakBench, MaliciousInstruct, and WildJailbreak adversarial_harmful.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.judge_outputs_llm_asr import _summarize_ahs  # noqa: E402
from scripts.recompute_saved_benchmark_scores import (  # noqa: E402
    DEFAULT_HARMBENCH_METADATA_PATH,
    _attach_harmbench_category,
    _harmbench_indexes,
)
from utils.saved_benchmark_postprocess import (  # noqa: E402
    judgement_summary_markdown,
    load_judgements,
    normalize_prompt,
    row_prompt,
    summarize_judgements,
    write_csv,
    write_json,
    write_jsonl,
)

RESULTS_DIR_NAME = "benign_sample_audit"
AUDIT_REPORT_VERSION = 2
AHS_AUDIT_REPORT_VERSION = 1
AHS_PAPER_BENCHMARKS = (
    "advbench",
    "harmbench-standard",
    "jailbreakbench",
    "malicious_instruct",
    "wildjailbreak-ah",
)
BENIGN_LABEL_MARKERS = ("benign", "safe", "harmless", "allowed")
HARMFUL_LABEL_MARKERS = ("harmful", "unsafe", "dangerous", "disallowed")
_PROMPT_COLUMNS = ("prompt", "adversarial", "instruction", "goal", "text")
_LABEL_COLUMNS = ("label", "data_type", "safety_label", "classification")
_ID_COLUMNS = ("id", "original_id", "sample_id", "index")


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    row_index: int
    sample_id: str
    prompt: str
    label: str
    source_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit saved RIGJ LLM-ASR/AHS rows against official HarmBench and "
            "WildJailbreak metadata."
        )
    )
    parser.add_argument(
        "--judge-output-path",
        type=Path,
        required=True,
        help=(
            "A *.summary.json / *.judgements.jsonl / *.inputs.jsonl file, a "
            "judge directory, or a RIGJ run directory."
        ),
    )
    parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to <judge-dir>/benign_sample_audit (or the input "
            "file's parent/benign_sample_audit)."
        ),
    )
    parser.add_argument(
        "--audit-run-name",
        default=None,
        help="Output file stem; inferred from the judgement file or input path.",
    )
    parser.add_argument(
        "--harmbench-metadata",
        type=Path,
        default=DEFAULT_HARMBENCH_METADATA_PATH,
        help=(
            "Official HarmBench text_test CSV/JSONL. It must contain exactly "
            "159 standard rows."
        ),
    )
    parser.add_argument(
        "--wildjailbreak-arrow",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Original WildJailbreak Arrow file(s). When omitted, local RIGJ "
            "and Hugging Face dataset caches are searched."
        ),
    )
    parser.add_argument(
        "--include-inputs-jsonl",
        action="store_true",
        help="Also discover *.inputs.jsonl when the input is a directory.",
    )
    parser.add_argument(
        "--fail-on-unmatched-wildjailbreak",
        action="store_true",
        help="Write all reports, then exit non-zero if WildJailbreak rows are unknown.",
    )
    return parser


def prompt_hash(value: Any) -> str:
    return hashlib.sha256(normalize_prompt(value).encode("utf-8")).hexdigest()


def label_is_benign(label: Any) -> bool | None:
    lowered = str(label or "").strip().lower()
    if not lowered:
        return None
    if any(marker in lowered for marker in BENIGN_LABEL_MARKERS):
        return True
    if any(marker in lowered for marker in HARMFUL_LABEL_MARKERS):
        return False
    return None


def _default_output_dir(input_path: Path) -> Path:
    input_path = Path(input_path)
    if input_path.is_file():
        return input_path.parent / RESULTS_DIR_NAME
    judge_dir = input_path / "llm_asr_judge"
    return (judge_dir if judge_dir.is_dir() else input_path) / RESULTS_DIR_NAME


def _resolve_judge_output_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    if input_path.is_file() and input_path.name.endswith(".summary.json"):
        judgement_path = input_path.with_name(
            input_path.name[: -len(".summary.json")] + ".judgements.jsonl"
        )
        if not judgement_path.is_file():
            raise FileNotFoundError(
                f"judgements corresponding to {input_path} not found: "
                f"{judgement_path}"
            )
        return judgement_path
    return input_path


def _default_run_name(input_path: Path, paths: list[Path]) -> str:
    if input_path.is_file():
        candidate = input_path.name
    else:
        judgement_paths = [
            path for path in paths if path.name.endswith(".judgements.jsonl")
        ]
        candidate = (
            judgement_paths[0].name if len(judgement_paths) == 1 else input_path.name
        )
    for suffix in (".judgements.jsonl", ".inputs.jsonl", ".jsonl"):
        if candidate.endswith(suffix):
            return candidate[: -len(suffix)]
    return candidate


def _candidate_cache_roots() -> list[Path]:
    roots = [
        ROOT_DIR / "outputs" / "benchmark_cache" / "audit_sources",
        ROOT_DIR / "outputs" / "hf_cache",
        ROOT_DIR / ".cache" / "huggingface" / "datasets",
        Path.home() / ".cache" / "huggingface" / "datasets",
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def discover_wildjailbreak_arrow_paths() -> list[Path]:
    matches: set[Path] = set()
    for root in _candidate_cache_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("*.arrow"):
            lowered = path.as_posix().lower()
            if "wild" in lowered and "jailbreak" in lowered:
                matches.add(path.resolve())
    return sorted(matches)


def _arrow_table(path: Path) -> Any:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required only when WildJailbreak Arrow files are read; "
            "install pyarrow or run without Arrow data to preserve rows as unknown"
        ) from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_file(source).read_all()


def _first_column(column_names: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {name.lower(): name for name in column_names}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def read_wildjailbreak_arrow(path: Path) -> list[SourceRecord]:
    table = _arrow_table(path)
    names = list(table.column_names)
    prompt_column = _first_column(names, _PROMPT_COLUMNS)
    label_column = _first_column(names, _LABEL_COLUMNS)
    id_column = _first_column(names, _ID_COLUMNS)
    if prompt_column is None or label_column is None:
        return []
    prompts = table.column(prompt_column).to_pylist()
    labels = table.column(label_column).to_pylist()
    ids = (
        table.column(id_column).to_pylist()
        if id_column is not None
        else [None] * len(prompts)
    )
    records: list[SourceRecord] = []
    for index, (prompt, label, raw_id) in enumerate(zip(prompts, labels, ids)):
        prompt_text = normalize_prompt(prompt)
        label_text = str(label or "").strip()
        if not prompt_text or not label_text:
            continue
        records.append(
            SourceRecord(
                dataset="wildjailbreak",
                row_index=index,
                sample_id=(
                    str(raw_id)
                    if raw_id is not None and str(raw_id).strip()
                    else f"wildjailbreak-{index:05d}"
                ),
                prompt=prompt_text,
                label=label_text,
                source_path=str(Path(path).resolve()),
            )
        )
    return records


def build_source_indexes(
    paths: list[Path],
) -> tuple[
    dict[str, list[SourceRecord]],
    dict[str, list[SourceRecord]],
    dict[str, Counter[str]],
]:
    by_id: dict[str, list[SourceRecord]] = defaultdict(list)
    by_prompt: dict[str, list[SourceRecord]] = defaultdict(list)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for record in read_wildjailbreak_arrow(path):
            key = (record.sample_id, prompt_hash(record.prompt), record.label)
            if key in seen:
                continue
            seen.add(key)
            by_id[record.sample_id].append(record)
            by_prompt[normalize_prompt(record.prompt)].append(record)
            label_counts[record.dataset][record.label] += 1
    return dict(by_id), dict(by_prompt), dict(label_counts)


def _pick_unique_label(records: list[SourceRecord]) -> SourceRecord | None:
    usable = [record for record in records if label_is_benign(record.label) is not None]
    classifications = {label_is_benign(record.label) for record in usable}
    return usable[0] if usable and len(classifications) == 1 else None


def match_wildjailbreak_row(
    row: dict[str, Any],
    *,
    by_id: dict[str, list[SourceRecord]],
    by_prompt: dict[str, list[SourceRecord]],
) -> tuple[SourceRecord | None, str, list[str]]:
    prompt = normalize_prompt(row_prompt(row))
    notes: list[str] = []
    id_candidates = [
        str(row.get("original_id") or "").strip(),
        str(row.get("sample_id") or "").strip(),
    ]
    for candidate_id in id_candidates:
        if not candidate_id:
            continue
        # RIGJ sample ids may be regenerated after harmful-only filtering, so an
        # id is trusted only when the original prompt also agrees exactly.
        candidates = [
            record
            for record in by_id.get(candidate_id, [])
            if normalize_prompt(record.prompt) == prompt
        ]
        match = _pick_unique_label(candidates)
        if match is not None:
            return match, "arrow_id_and_prompt_exact", notes
        if candidates:
            notes.append(f"Arrow id {candidate_id!r} has conflicting labels")

    prompt_candidates = by_prompt.get(prompt, [])
    match = _pick_unique_label(prompt_candidates)
    if match is not None:
        return match, "arrow_prompt_exact", notes
    if prompt_candidates:
        notes.append("Arrow prompt has conflicting or unrecognized labels")
    else:
        notes.append("no Arrow prompt match")
    return None, "unmatched", notes


def annotate_row(
    row: dict[str, Any],
    *,
    by_id: dict[str, list[SourceRecord]],
    by_prompt: dict[str, list[SourceRecord]],
) -> dict[str, Any]:
    annotated = dict(row)
    prompt = row_prompt(row)
    annotated["benign_audit_prompt_sha256"] = prompt_hash(prompt)
    if str(row.get("benchmark") or "").strip().lower() != "wildjailbreak":
        annotated.update(
            {
                "benign_audit_status": "not_applicable",
                "benign_audit_label": None,
                "benign_audit_is_benign": None,
                "benign_audit_is_harmful": None,
                "benign_audit_match_method": "not_applicable",
                "benign_audit_match_dataset": None,
                "benign_audit_match_row_index": None,
                "benign_audit_match_source_path": None,
                "benign_audit_notes": [],
            }
        )
        return annotated

    match, method, notes = match_wildjailbreak_row(
        row, by_id=by_id, by_prompt=by_prompt
    )
    is_benign = label_is_benign(match.label) if match is not None else None
    annotated.update(
        {
            "benign_audit_status": (
                "benign"
                if is_benign is True
                else "harmful" if is_benign is False else "unknown"
            ),
            "benign_audit_label": match.label if match is not None else None,
            "benign_audit_is_benign": is_benign,
            "benign_audit_is_harmful": (None if is_benign is None else not is_benign),
            "benign_audit_match_method": method,
            "benign_audit_match_dataset": match.dataset if match else None,
            "benign_audit_match_row_index": match.row_index if match else None,
            "benign_audit_match_source_path": match.source_path if match else None,
            "benign_audit_notes": notes,
        }
    )
    return annotated


def audit_summary(
    rows: list[dict[str, Any]],
    *,
    label_counts: dict[str, Counter[str]],
) -> dict[str, Any]:
    wild_rows = [
        row
        for row in rows
        if str(row.get("benchmark") or "").strip().lower() == "wildjailbreak"
    ]
    statuses = Counter(str(row.get("benign_audit_status")) for row in wild_rows)
    methods = Counter(str(row.get("benign_audit_match_method")) for row in wild_rows)
    labels = Counter(
        str(row.get("benign_audit_label"))
        for row in wild_rows
        if row.get("benign_audit_label") is not None
    )
    by_job: dict[str, Counter[str]] = defaultdict(Counter)
    by_pair: dict[str, Counter[str]] = defaultdict(Counter)
    for row in wild_rows:
        status = str(row.get("benign_audit_status") or "unknown")
        by_job[str(row.get("job_id") or "unknown")][status] += 1
        by_job[str(row.get("job_id") or "unknown")]["rows"] += 1
        by_pair[str(row.get("pair_id") or "unknown")][status] += 1
        by_pair[str(row.get("pair_id") or "unknown")]["rows"] += 1
    return {
        "report_version": AUDIT_REPORT_VERSION,
        "all_judgement_rows": len(rows),
        "wildjailbreak_rows": len(wild_rows),
        "wildjailbreak_benign_rows": statuses["benign"],
        "wildjailbreak_harmful_rows": statuses["harmful"],
        "wildjailbreak_unknown_rows": statuses["unknown"],
        "status_counts": dict(statuses),
        "match_methods": dict(methods),
        "matched_labels": dict(labels),
        "source_label_counts": {
            key: dict(value) for key, value in sorted(label_counts.items())
        },
        "by_generation_job": {
            key: dict(value) for key, value in sorted(by_job.items())
        },
        "by_pair": {key: dict(value) for key, value in sorted(by_pair.items())},
    }


def corrected_asr_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded = [
        row
        for row in rows
        if str(row.get("benchmark") or "").strip().lower() == "wildjailbreak"
        and row.get("benign_audit_is_benign") is True
    ]
    corrected = [
        row
        for row in rows
        if not (
            str(row.get("benchmark") or "").strip().lower() == "wildjailbreak"
            and row.get("benign_audit_is_benign") is True
        )
    ]
    wild = [
        row
        for row in rows
        if str(row.get("benchmark") or "").strip().lower() == "wildjailbreak"
    ]
    return corrected, {
        "excluded_wildjailbreak_benign_rows": len(excluded),
        "included_wildjailbreak_harmful_rows": sum(
            row.get("benign_audit_is_benign") is False for row in wild
        ),
        "included_wildjailbreak_unknown_rows": sum(
            row.get("benign_audit_is_benign") is None for row in wild
        ),
    }


def _audit_markdown(
    summary: dict[str, Any],
    corrected_summary: dict[str, Any],
    wildjailbreak_summary: dict[str, Any],
) -> str:
    lines = [
        "# WildJailbreak Benign-Sample Audit",
        "",
        "## Audit overview",
        "",
        "| all rows | WildJailbreak | benign excluded | harmful retained | unknown retained |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['all_judgement_rows']} | {summary['wildjailbreak_rows']} | "
            f"{summary['wildjailbreak_benign_rows']} | "
            f"{summary['wildjailbreak_harmful_rows']} | "
            f"{summary['wildjailbreak_unknown_rows']} |"
        ),
        "",
        "## Arrow sources",
        "",
    ]
    arrow_paths = summary.get("source_files", {}).get("wildjailbreak_arrow", [])
    if arrow_paths:
        lines.extend(f"- `{path}`" for path in arrow_paths)
    else:
        lines.append("- None found; WildJailbreak rows are retained as `unknown`.")
    wildjailbreak_md = judgement_summary_markdown(
        wildjailbreak_summary,
        title="WildJailbreak-Only adversarial_harmful ASR",
    )
    corrected_md = judgement_summary_markdown(
        corrected_summary,
        title="All-Benchmark ASR After WildJailbreak Benign Exclusion",
    )
    lines.extend(
        [
            "",
            wildjailbreak_md.rstrip(),
            "",
            corrected_md.rstrip(),
            "",
        ]
    )
    return "\n".join(lines)


def _judgement_metric(row: dict[str, Any]) -> str:
    explicit = str(row.get("judge_metric") or "").strip().lower()
    if explicit in {"asr", "ahs"}:
        return explicit
    if "harmfulness_score" in row or "ahs_score" in row:
        return "ahs"
    return "asr"


def filter_ahs_paper_five_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    input_benchmarks: Counter[str] = Counter()
    retained_benchmarks: Counter[str] = Counter()
    excluded_reasons: Counter[str] = Counter()
    harmbench_categories: Counter[str] = Counter()
    wildjailbreak_labels: Counter[str] = Counter()
    allowed_direct = {"advbench", "jailbreakbench", "malicious_instruct"}

    for raw_row in rows:
        row = dict(raw_row)
        benchmark = str(row.get("benchmark") or "").strip().lower()
        input_benchmarks[benchmark or "unknown"] += 1
        report_benchmark: str | None = None
        if benchmark in allowed_direct:
            report_benchmark = benchmark
        elif benchmark == "harmbench":
            category = str(row.get("harmbench_category") or "unknown").lower()
            harmbench_categories[category] += 1
            if category == "standard":
                report_benchmark = "harmbench-standard"
            else:
                excluded_reasons[f"harmbench_{category}"] += 1
        elif benchmark == "wildjailbreak":
            label = str(row.get("benign_audit_label") or "unknown").lower()
            wildjailbreak_labels[label] += 1
            if label == "adversarial_harmful":
                report_benchmark = "wildjailbreak-ah"
            elif label == "unknown":
                excluded_reasons["wildjailbreak_unknown"] += 1
            else:
                excluded_reasons[f"wildjailbreak_{label}"] += 1
        else:
            excluded_reasons[f"benchmark_{benchmark or 'unknown'}"] += 1

        if report_benchmark is None:
            continue
        row["judge_benchmark"] = report_benchmark
        row["ahs_post_audit_included"] = True
        row["ahs_post_audit_benchmark"] = report_benchmark
        retained.append(row)
        retained_benchmarks[report_benchmark] += 1

    audit = {
        "input_row_count": len(rows),
        "retained_row_count": len(retained),
        "excluded_row_count": len(rows) - len(retained),
        "input_benchmark_counts": dict(sorted(input_benchmarks.items())),
        "retained_benchmark_counts": dict(sorted(retained_benchmarks.items())),
        "excluded_reason_counts": dict(sorted(excluded_reasons.items())),
        "harmbench_category_counts": dict(sorted(harmbench_categories.items())),
        "wildjailbreak_label_counts": dict(sorted(wildjailbreak_labels.items())),
    }
    return retained, audit


def _format_ahs(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _format_ahs_group(group: dict[str, Any]) -> str:
    return f"{group['judged_count']}/{group['selected_count']}"


def ahs_paper_five_markdown(
    summary: dict[str, Any],
    *,
    title: str = "AHS Paper-Five Post-Audit",
) -> str:
    audit = dict(summary.get("filter_audit") or {})
    lines = [
        f"# {title}",
        "",
        "> Only AdvBench, official HarmBench standard, JailbreakBench, "
        "MaliciousInstruct (MI), and WildJailbreak adversarial_harmful (AH) "
        "participate in the final AHS. All other rows are excluded before "
        "aggregation.",
        "",
        "## Filter audit",
        "",
        "| input rows | retained rows | excluded rows | judged/retained | judge errors | AHS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {audit.get('input_row_count', 0)} | "
            f"{audit.get('retained_row_count', 0)} | "
            f"{audit.get('excluded_row_count', 0)} | "
            f"{summary['overall']['judged_count']}/{summary['overall']['selected_count']} | "
            f"{summary['overall']['judge_error_count']} | "
            f"{_format_ahs(summary['overall']['ahs'])} |"
        ),
        "",
        "### Retained benchmark totals",
        "",
        "| benchmark | selected n | judged/n | errors | AHS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for benchmark in AHS_PAPER_BENCHMARKS:
        group = dict(summary.get("by_benchmark", {}).get(benchmark) or {})
        if not group:
            continue
        lines.append(
            f"| {benchmark} | {group['selected_count']} | "
            f"{_format_ahs_group(group)} | {group['judge_error_count']} | "
            f"{_format_ahs(group['ahs'])} |"
        )

    lines.extend(
        [
            "",
            "## Pair × benchmark AHS detail",
            "",
            "| pair | benchmark | source judged/n | source errors | source AHS | "
            "target judged/n | target errors | target AHS | AHS gap (target-source) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair_id, pair in sorted(dict(summary.get("by_pair") or {}).items()):
        groups = [("overall", pair)]
        groups.extend(
            (benchmark, pair.get("by_benchmark", {}).get(benchmark, {}))
            for benchmark in AHS_PAPER_BENCHMARKS
        )
        for benchmark, value in groups:
            source = dict(value.get("source") or {})
            target = dict(value.get("target") or {})
            if not source or not target:
                continue
            lines.append(
                f"| {pair_id} | {benchmark} | {_format_ahs_group(source)} | "
                f"{source['judge_error_count']} | {_format_ahs(source['ahs'])} | "
                f"{_format_ahs_group(target)} | {target['judge_error_count']} | "
                f"{_format_ahs(target['ahs'])} | "
                f"{_format_ahs(value.get('ahs_gap'))} |"
            )

    excluded = dict(audit.get("excluded_reason_counts") or {})
    lines.extend(
        [
            "",
            "## Exclusion audit",
            "",
            "| exclusion reason | rows |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| {reason} | {count} |" for reason, count in sorted(excluded.items())
    )
    lines.extend(
        [
            "",
            "## Sources",
            "",
            *[
                f"- judgement: `{path}`"
                for path in summary.get("source_files", {}).get("judge_jsonl", [])
            ],
            f"- HarmBench metadata: `{summary.get('source_files', {}).get('harmbench_metadata')}`",
            *[
                f"- WildJailbreak Arrow: `{path}`"
                for path in summary.get("source_files", {}).get(
                    "wildjailbreak_arrow", []
                )
            ],
            "",
        ]
    )
    return "\n".join(lines)


def run_ahs_paper_five_audit(
    args: argparse.Namespace,
    *,
    input_path: Path,
    rows: list[dict[str, Any]],
    jsonl_paths: list[Path],
) -> dict[str, Any]:
    arrow_paths = (
        [Path(path).resolve() for path in args.wildjailbreak_arrow]
        if args.wildjailbreak_arrow is not None
        else discover_wildjailbreak_arrow_paths()
    )
    if not arrow_paths:
        raise FileNotFoundError(
            "no WildJailbreak Arrow metadata found; pass --wildjailbreak-arrow"
        )
    by_id, by_prompt, label_counts = build_source_indexes(arrow_paths)
    harmbench_by_id, harmbench_by_prompt, harmbench_metadata = _harmbench_indexes(args)

    annotated: list[dict[str, Any]] = []
    for row in rows:
        value = annotate_row(row, by_id=by_id, by_prompt=by_prompt)
        if str(row.get("benchmark") or "").strip().lower() == "harmbench":
            value = _attach_harmbench_category(
                value,
                by_id=harmbench_by_id,
                by_prompt=harmbench_by_prompt,
            )
        annotated.append(value)
    filtered_rows, filter_audit = filter_ahs_paper_five_rows(annotated)
    summary = _summarize_ahs(filtered_rows)
    summary.update(
        {
            "report_version": AHS_AUDIT_REPORT_VERSION,
            "metric_name": "AHS paper-five metadata-filtered post-audit",
            "filter_audit": filter_audit,
            "wildjailbreak_source_label_counts": {
                key: dict(value) for key, value in sorted(label_counts.items())
            },
            "harmbench_metadata": harmbench_metadata,
            "source_files": {
                "judge_jsonl": [str(path.resolve()) for path in jsonl_paths],
                "harmbench_metadata": harmbench_metadata["path"],
                "wildjailbreak_arrow": [str(path) for path in arrow_paths],
            },
        }
    )

    output_dir = Path(args.audit_output_dir or _default_output_dir(input_path))
    run_name = args.audit_run_name or _default_run_name(input_path, jsonl_paths)
    annotated_path = output_dir / f"{run_name}.ahs_paper5.annotated.jsonl"
    rows_path = output_dir / f"{run_name}.ahs_paper5.rows.jsonl"
    csv_path = output_dir / f"{run_name}.ahs_paper5.rows.csv"
    summary_path = output_dir / f"{run_name}.ahs_paper5.summary.json"
    markdown_path = output_dir / f"{run_name}.ahs_paper5.summary.md"
    write_jsonl(annotated_path, annotated)
    write_jsonl(rows_path, filtered_rows)
    write_csv(
        csv_path,
        filtered_rows,
        preferred_fields=(
            "judge_sample_key",
            "job_id",
            "pair_id",
            "model_role",
            "benchmark",
            "judge_benchmark",
            "sample_id",
            "original_id",
            "goal",
            "harmbench_category",
            "benign_audit_label",
            "harmfulness_score",
            "judge_error",
        ),
    )
    write_json(summary_path, summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(ahs_paper_five_markdown(summary), encoding="utf-8")

    print(
        "[ahs-paper-five] "
        f"input={filter_audit['input_row_count']} "
        f"retained={filter_audit['retained_row_count']} "
        f"excluded={filter_audit['excluded_row_count']} "
        f"judged={summary['overall']['judged_count']} "
        f"errors={summary['overall']['judge_error_count']} "
        f"ahs={summary['overall']['ahs']}"
    )
    print(f"[done] summary={markdown_path}")
    unknown_count = filter_audit["excluded_reason_counts"].get(
        "wildjailbreak_unknown", 0
    )
    if args.fail_on_unmatched_wildjailbreak and unknown_count:
        raise SystemExit(
            f"unmatched WildJailbreak rows: {unknown_count} (reports were written)"
        )
    return summary


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    input_path = _resolve_judge_output_path(Path(args.judge_output_path))
    rows, jsonl_paths = load_judgements(
        input_path, include_inputs=bool(args.include_inputs_jsonl)
    )
    metrics = {_judgement_metric(row) for row in rows}
    if metrics == {"ahs"}:
        return run_ahs_paper_five_audit(
            args,
            input_path=input_path,
            rows=rows,
            jsonl_paths=jsonl_paths,
        )
    if metrics != {"asr"}:
        raise ValueError(f"mixed or unsupported judgement metrics: {sorted(metrics)}")
    arrow_paths = (
        [Path(path).resolve() for path in args.wildjailbreak_arrow]
        if args.wildjailbreak_arrow is not None
        else discover_wildjailbreak_arrow_paths()
    )
    by_id, by_prompt, label_counts = build_source_indexes(arrow_paths)
    annotated = [annotate_row(row, by_id=by_id, by_prompt=by_prompt) for row in rows]
    summary = audit_summary(annotated, label_counts=label_counts)
    corrected_rows, correction = corrected_asr_rows(annotated)
    # Original manifests describe the pre-audit selection and therefore must
    # not be reused as denominators after benign rows have been removed.
    corrected_summary = summarize_judgements(corrected_rows)
    corrected_wildjailbreak_rows = [
        row
        for row in corrected_rows
        if str(row.get("benchmark") or "").strip().lower() == "wildjailbreak"
    ]
    wildjailbreak_summary = summarize_judgements(corrected_wildjailbreak_rows)
    wildjailbreak_summary.update(
        {
            "report_version": AUDIT_REPORT_VERSION,
            "metric_name": ("WildJailbreak-only adversarial_harmful common LLM ASR"),
            "wildjailbreak_asr_correction": correction,
        }
    )
    corrected_summary.update(
        {
            "report_version": AUDIT_REPORT_VERSION,
            "metric_name": "WildJailbreak-benign-excluded common LLM ASR",
            "wildjailbreak_asr_correction": correction,
            "wildjailbreak_only": wildjailbreak_summary,
        }
    )
    summary.update(
        {
            "source_files": {
                "judge_jsonl": [str(path.resolve()) for path in jsonl_paths],
                "wildjailbreak_arrow": [str(path) for path in arrow_paths],
            },
            "asr_correction": correction,
            "corrected_overall": corrected_summary["overall"],
            "corrected_wildjailbreak_overall": wildjailbreak_summary["overall"],
        }
    )

    output_dir = Path(args.audit_output_dir or _default_output_dir(input_path))
    run_name = args.audit_run_name or _default_run_name(input_path, jsonl_paths)
    annotated_path = output_dir / f"{run_name}.annotated.jsonl"
    benign_path = output_dir / f"{run_name}.benign.jsonl"
    corrected_path = output_dir / f"{run_name}.asr_corrected.rows.jsonl"
    wildjailbreak_path = output_dir / f"{run_name}.wildjailbreak_only.rows.jsonl"
    write_jsonl(annotated_path, annotated)
    write_jsonl(
        benign_path,
        [row for row in annotated if row.get("benign_audit_is_benign") is True],
    )
    write_jsonl(corrected_path, corrected_rows)
    write_jsonl(wildjailbreak_path, corrected_wildjailbreak_rows)
    write_csv(
        output_dir / f"{run_name}.annotated.csv",
        annotated,
        preferred_fields=(
            "judge_sample_key",
            "job_id",
            "pair_id",
            "model_role",
            "benchmark",
            "sample_id",
            "original_id",
            "goal",
            "benign_audit_status",
            "benign_audit_label",
            "benign_audit_match_method",
            "any_harmful_output",
            "prompt_matched_attack_success",
            "judge_error",
        ),
    )
    write_csv(
        output_dir / f"{run_name}.asr_corrected.rows.csv",
        corrected_rows,
        preferred_fields=(
            "judge_sample_key",
            "job_id",
            "pair_id",
            "model_role",
            "benchmark",
            "sample_id",
            "goal",
            "benign_audit_status",
            "prompt_matched_attack_success",
        ),
    )
    write_csv(
        output_dir / f"{run_name}.wildjailbreak_only.rows.csv",
        corrected_wildjailbreak_rows,
        preferred_fields=(
            "judge_sample_key",
            "job_id",
            "pair_id",
            "model_role",
            "benchmark",
            "sample_id",
            "goal",
            "benign_audit_status",
            "prompt_matched_attack_success",
        ),
    )
    write_json(output_dir / f"{run_name}.summary.json", summary)
    write_json(output_dir / f"{run_name}.asr_corrected.summary.json", corrected_summary)
    write_json(
        output_dir / f"{run_name}.wildjailbreak_only.summary.json",
        wildjailbreak_summary,
    )
    wildjailbreak_markdown_path = (
        output_dir / f"{run_name}.wildjailbreak_only.summary.md"
    )
    wildjailbreak_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    wildjailbreak_markdown_path.write_text(
        judgement_summary_markdown(
            wildjailbreak_summary,
            title="WildJailbreak-Only adversarial_harmful ASR",
        ),
        encoding="utf-8",
    )
    markdown_path = output_dir / f"{run_name}.asr_corrected.summary.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _audit_markdown(summary, corrected_summary, wildjailbreak_summary),
        encoding="utf-8",
    )

    print(f"[audit] rows={len(annotated)} arrow_files={len(arrow_paths)}")
    print(
        "[wildjailbreak] "
        f"rows={summary['wildjailbreak_rows']} "
        f"benign={summary['wildjailbreak_benign_rows']} "
        f"harmful={summary['wildjailbreak_harmful_rows']} "
        f"unknown={summary['wildjailbreak_unknown_rows']}"
    )
    print("[wildjailbreak-only-asr-corrected-summary]")
    print(
        "pair\tsource_retained\tsource_judged\tsource_asr\t"
        "target_retained\ttarget_judged\ttarget_asr\tgap"
    )
    for pair_id, value in wildjailbreak_summary.get("by_pair", {}).items():
        print(
            f"{pair_id}\t{value['source']['selected_count']}\t"
            f"{value['source']['judged_count']}\t{value['source']['asr']}\t"
            f"{value['target']['selected_count']}\t"
            f"{value['target']['judged_count']}\t{value['target']['asr']}\t"
            f"{value['asr_gap']}"
        )
    print(f"[done] summary={markdown_path}")
    if args.fail_on_unmatched_wildjailbreak and summary["wildjailbreak_unknown_rows"]:
        raise SystemExit(
            "unmatched WildJailbreak rows: "
            f"{summary['wildjailbreak_unknown_rows']} (reports were written)"
        )
    return summary


def main() -> None:
    run_audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
