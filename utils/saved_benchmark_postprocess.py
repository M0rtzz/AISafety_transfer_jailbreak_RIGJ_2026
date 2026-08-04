"""Shared readers, deduplication and summaries for saved RIGJ evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from utils.benchmark_io import stable_hash, utc_timestamp

_SKIP_PARTS = {
    "benign_sample_audit",
    "recomputed_benchmark_scores",
}


def normalize_prompt(value: Any) -> str:
    return " ".join(str(value or "").split())


def row_prompt(row: dict[str, Any]) -> str:
    return str(
        row.get("goal")
        or row.get("prompt")
        or row.get("instruction")
        or row.get("behavior")
        or row.get("Behavior")
        or ""
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    preferred_fields: Iterable[str] = (),
) -> None:
    preferred = list(dict.fromkeys(str(field) for field in preferred_fields))
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + remaining
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def load_job_manifests(input_path: Path) -> dict[str, dict[str, Any]]:
    input_path = Path(input_path)
    candidates = (
        [input_path.parent / "job_manifest.json"]
        if input_path.is_file()
        else sorted(input_path.rglob("job_manifest.json"))
    )
    manifests: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.is_file() or _SKIP_PARTS.intersection(path.parts):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("job_id"):
            manifests[str(value["job_id"])] = value
    return manifests


def manifest_selected_count(
    manifests: dict[str, dict[str, Any]],
    *,
    job_ids: set[str] | None = None,
    benchmark: str | None = None,
) -> int | None:
    if not manifests:
        return None
    total = 0
    matched = False
    for job_id, manifest in manifests.items():
        if job_ids is not None and job_id not in job_ids:
            continue
        for bench_name, summary in dict(manifest.get("summaries") or {}).items():
            if benchmark is not None and str(bench_name) != benchmark:
                continue
            total += int(dict(summary).get("selected_count", 0))
            matched = True
    return total if matched else None


def _judgement_paths(path: Path, *, include_inputs: bool) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    patterns = ["*.judgements.jsonl"]
    if include_inputs:
        patterns.append("*.inputs.jsonl")
    paths: set[Path] = set()
    for pattern in patterns:
        for candidate in path.rglob(pattern):
            if not _SKIP_PARTS.intersection(candidate.parts):
                paths.add(candidate)
    return sorted(paths)


def judgement_key(row: dict[str, Any]) -> str:
    explicit = row.get("judge_sample_key") or row.get("generation_key")
    if explicit:
        return str(explicit)
    return stable_hash(
        {
            key: row.get(key)
            for key in (
                "job_id",
                "pair_id",
                "benchmark",
                "sample_id",
                "goal",
                "prompt",
                "model_output",
            )
        },
        length=32,
    )


def judgement_is_successful(row: dict[str, Any]) -> bool:
    return (
        not str(row.get("judge_error") or "").strip()
        and isinstance(row.get("any_harmful_output"), bool)
        and isinstance(row.get("prompt_matched_attack_success"), bool)
    )


def load_judgements(
    path: Path,
    *,
    include_inputs: bool = False,
) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = _judgement_paths(path, include_inputs=include_inputs)
    if not paths:
        raise FileNotFoundError(f"no judge JSONL files found under {path}")
    by_key: dict[str, dict[str, Any]] = {}
    source_by_key: dict[str, Path] = {}
    for jsonl_path in paths:
        for raw in read_jsonl(jsonl_path):
            row = {**raw, "audit_input_jsonl": str(jsonl_path.resolve())}
            key = judgement_key(row)
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = row
                source_by_key[key] = jsonl_path
                continue
            same_input = normalize_prompt(row_prompt(previous)) == normalize_prompt(
                row_prompt(row)
            ) and str(previous.get("model_output") or "") == str(
                row.get("model_output") or ""
            )
            if not same_input:
                raise ValueError(
                    f"conflicting duplicate judgement input for key {key}: "
                    f"{source_by_key[key]} versus {jsonl_path}"
                )
            previous_success = judgement_is_successful(previous)
            current_success = judgement_is_successful(row)
            if previous_success and current_success:
                old_verdict = (
                    previous.get("any_harmful_output"),
                    previous.get("prompt_matched_attack_success"),
                )
                new_verdict = (
                    row.get("any_harmful_output"),
                    row.get("prompt_matched_attack_success"),
                )
                if old_verdict != new_verdict:
                    raise ValueError(
                        f"conflicting duplicate judgement verdict for key {key}: "
                        f"{source_by_key[key]} has {old_verdict}, "
                        f"{jsonl_path} has {new_verdict}"
                    )
                continue
            if previous_success and not current_success:
                continue
            by_key[key] = row
            source_by_key[key] = jsonl_path
    rows = list(by_key.values())
    rows.sort(
        key=lambda row: (
            str(row.get("attack_source_family") or ""),
            str(row.get("eval_model_family") or ""),
            str(row.get("benchmark") or ""),
            str(row.get("sample_id") or ""),
        )
    )
    return rows, paths


def generation_files(run_dir: Path, benchmark: str) -> list[Path]:
    run_dir = Path(run_dir)
    if run_dir.is_file():
        return [run_dir]
    generation_root = run_dir / "generations"
    root = generation_root if generation_root.is_dir() else run_dir
    return sorted(
        path
        for path in root.rglob(f"{benchmark}.jsonl")
        if not _SKIP_PARTS.intersection(path.parts)
        and "llm_asr_judge" not in path.parts
    )


def load_generation_rows(
    run_dir: Path,
    *,
    benchmark: str,
    model_roles: set[str] | None = None,
    job_ids: set[str] | None = None,
    pair_ids: set[str] | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for path in generation_files(run_dir, benchmark):
        for row in read_jsonl(path):
            if row.get("record_type") != "rigj_harm_generation":
                continue
            if str(row.get("benchmark") or "").lower() != benchmark.lower():
                continue
            if row.get("status") != "success":
                continue
            if (
                not str(row_prompt(row)).strip()
                or not str(row.get("model_output") or "").strip()
            ):
                continue
            if model_roles and str(row.get("model_role")) not in model_roles:
                continue
            if job_ids and str(row.get("job_id")) not in job_ids:
                continue
            if pair_ids and str(row.get("pair_id")) not in pair_ids:
                continue
            generation_key = str(
                row.get("generation_key") or stable_hash(row, length=32)
            )
            revision = row.get("generation_revision")
            key = (
                stable_hash(
                    {
                        "generation_key": generation_key,
                        "generation_revision": revision,
                        "model_output": row.get("model_output"),
                    },
                    length=32,
                )
                if revision
                else generation_key
            )
            unique[key] = {
                **row,
                "judge_sample_key": key,
                "generation_file": str(path.resolve()),
            }
    rows = list(unique.values())
    rows.sort(
        key=lambda row: (
            str(row.get("attack_source_family") or ""),
            str(row.get("eval_model_family") or ""),
            str(row.get("sample_id") or ""),
        )
    )
    if max_samples is not None:
        if int(max_samples) < 1:
            raise ValueError("--max-samples must be >= 1")
        rows = rows[: int(max_samples)]
    return rows


def group_summary(
    rows: list[dict[str, Any]],
    selected_count: int | None = None,
) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if isinstance(row.get("prompt_matched_attack_success"), bool)
    ]
    selected = int(selected_count if selected_count is not None else len(rows))
    matched = sum(bool(row["prompt_matched_attack_success"]) for row in successful)
    harmful = sum(bool(row.get("any_harmful_output")) for row in successful)
    judged = len(successful)
    return {
        "selected_count": selected,
        "generation_available_count": len(rows),
        "generation_unavailable_count": max(0, selected - len(rows)),
        "judged_count": judged,
        "judge_error_count": len(rows) - judged,
        "coverage": judged / selected if selected else 0.0,
        "prompt_matched_attack_success_count": matched,
        "prompt_matched_attack_success_rate": matched / judged if judged else None,
        "prompt_matched_attack_success_total_rate": (
            matched / selected if selected else None
        ),
        "asr": matched / judged if judged else None,
        "conservative_asr": matched / selected if selected else None,
        "any_harmful_output_count": harmful,
        "any_harmful_output_rate": harmful / judged if judged else None,
        "conservative_any_harmful_output_rate": (
            harmful / selected if selected else None
        ),
    }


def summarize_judgements(
    rows: list[dict[str, Any]],
    manifests: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifests = manifests or {}
    by_benchmark: dict[str, Any] = {}
    by_job: dict[str, Any] = {}
    by_role: dict[str, Any] = {}
    all_benchmarks = {str(row.get("benchmark")) for row in rows if row.get("benchmark")}
    for manifest in manifests.values():
        all_benchmarks.update(str(key) for key in dict(manifest.get("summaries") or {}))
    for benchmark in sorted(all_benchmarks):
        group = [row for row in rows if str(row.get("benchmark")) == benchmark]
        by_benchmark[benchmark] = group_summary(
            group,
            manifest_selected_count(manifests, benchmark=benchmark),
        )
    all_job_ids = {str(row.get("job_id")) for row in rows if row.get("job_id")} | set(
        manifests
    )
    for job_id in sorted(all_job_ids):
        group = [row for row in rows if str(row.get("job_id")) == job_id]
        meta = group[0] if group else manifests.get(job_id, {})
        source = meta.get("attack_source_family")
        target = meta.get("eval_model_family") or meta.get("eval_family")
        role = meta.get("model_role") or ("source" if source == target else "target")
        by_job[job_id] = {
            "attack_source_family": source,
            "eval_model_family": target,
            "model_role": role,
            **group_summary(
                group,
                manifest_selected_count(manifests, job_ids={job_id}),
            ),
        }
    for role in ("source", "target"):
        group = [row for row in rows if row.get("model_role") == role]
        role_job_ids = {
            job_id
            for job_id, value in by_job.items()
            if value.get("model_role") == role
        }
        by_role[role] = group_summary(
            group,
            manifest_selected_count(manifests, job_ids=role_job_ids),
        )

    target_pairs = {
        str(row.get("pair_id"))
        for row in rows
        if row.get("model_role") == "target" and "_to_" in str(row.get("pair_id"))
    }
    for manifest in manifests.values():
        source = manifest.get("attack_source_family")
        target = manifest.get("eval_family") or manifest.get("eval_model_family")
        if source and target and source != target:
            target_pairs.add(f"{source}_to_{target}")
    by_pair: dict[str, Any] = {}
    for pair_id in sorted(target_pairs):
        source_family = pair_id.split("_to_", 1)[0]
        source_rows = [
            row
            for row in rows
            if row.get("model_role") == "source"
            and str(row.get("attack_source_family")) == source_family
        ]
        target_rows = [row for row in rows if str(row.get("pair_id")) == pair_id]
        source_job_ids = {
            job_id
            for job_id, value in by_job.items()
            if value.get("model_role") == "source"
            and value.get("attack_source_family") == source_family
        }
        target_job_ids = {
            job_id
            for job_id, value in by_job.items()
            if value.get("model_role") == "target"
            and f"{value.get('attack_source_family')}_to_{value.get('eval_model_family')}"
            == pair_id
        }
        source_summary = group_summary(
            source_rows,
            manifest_selected_count(manifests, job_ids=source_job_ids),
        )
        target_summary = group_summary(
            target_rows,
            manifest_selected_count(manifests, job_ids=target_job_ids),
        )
        source_asr = source_summary["asr"]
        target_asr = target_summary["asr"]
        by_pair[pair_id] = {
            "source": source_summary,
            "target": target_summary,
            "asr_gap": (
                target_asr - source_asr
                if source_asr is not None and target_asr is not None
                else None
            ),
            "target_source_asr_ratio": (
                target_asr / source_asr
                if source_asr not in (None, 0) and target_asr is not None
                else None
            ),
        }
    return {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "overall": group_summary(rows, manifest_selected_count(manifests)),
        "by_benchmark": by_benchmark,
        "by_model_role": by_role,
        "by_generation_job": by_job,
        "by_pair": by_pair,
    }


def _rate(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _asr_cell(value: dict[str, Any]) -> str:
    return (
        f"{int(value.get('prompt_matched_attack_success_count', 0))}/"
        f"{int(value.get('judged_count', 0))} ({_rate(value.get('asr'))})"
    )


def judgement_summary_markdown(summary: dict[str, Any], *, title: str) -> str:
    lines = [f"# {title}", "", "## Overall", ""]
    overall = summary["overall"]
    lines.extend(
        [
            "| n | judged | prompt-matched ASR | ASR(any) | errors |",
            "| ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {overall['selected_count']} | {overall['judged_count']} | "
                f"{_asr_cell(overall)} | "
                f"{overall['any_harmful_output_count']}/{overall['judged_count']} "
                f"({_rate(overall['any_harmful_output_rate'])}) | "
                f"{overall['judge_error_count']} |"
            ),
            "",
            "## By benchmark",
            "",
            "| benchmark | n | judged | prompt-matched ASR | errors |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for benchmark, value in summary.get("by_benchmark", {}).items():
        lines.append(
            f"| {benchmark} | {value['selected_count']} | {value['judged_count']} | "
            f"{_asr_cell(value)} | {value['judge_error_count']} |"
        )
    lines.extend(
        [
            "",
            "## By generation job",
            "",
            "| job | source | eval | role | n | judged | prompt-matched ASR | errors |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for job_id, value in summary.get("by_generation_job", {}).items():
        lines.append(
            f"| `{job_id}` | {value.get('attack_source_family')} | "
            f"{value.get('eval_model_family')} | {value.get('model_role')} | "
            f"{value['selected_count']} | {value['judged_count']} | "
            f"{_asr_cell(value)} | {value['judge_error_count']} |"
        )
    lines.extend(
        [
            "",
            "## By model pair",
            "",
            "| pair | source ASR | target ASR | gap | ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair_id, value in summary.get("by_pair", {}).items():
        lines.append(
            f"| {pair_id} | {_asr_cell(value['source'])} | "
            f"{_asr_cell(value['target'])} | {_rate(value.get('asr_gap'))} | "
            f"{_rate(value.get('target_source_asr_ratio'))} |"
        )
    return "\n".join(lines) + "\n"
