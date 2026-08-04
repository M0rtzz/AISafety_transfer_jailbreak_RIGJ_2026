#!/usr/bin/env python3
"""Re-score saved RIGJ StrongREJECT outputs and filter HarmBench standard rows."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai import OpenAI  # noqa: E402

from utils.benchmark_io import stable_hash, utc_timestamp  # noqa: E402
from utils.saved_benchmark_postprocess import (  # noqa: E402
    append_jsonl,
    judgement_key,
    judgement_summary_markdown,
    load_generation_rows,
    load_job_manifests,
    load_judgements,
    normalize_prompt,
    read_jsonl,
    row_prompt,
    summarize_judgements,
    write_csv,
    write_json,
    write_jsonl,
)
from utils.strongreject import (  # noqa: E402
    RUBRICS,
    StrongRejectRubric,
    get_strongreject_rubric,
    parse_strongreject_judgement,
    score_strongreject,
)

RESULTS_DIR_NAME = "recomputed_benchmark_scores"
STRONGREJECT_REPORT_VERSION = 1
HARMBENCH_STANDARD_REPORT_VERSION = 2
_HARMBENCH_CATEGORIES = {"standard", "contextual", "copyright"}
_HARMBENCH_NOT_IN_OFFICIAL_TEST = "not_in_official_test"
HARMBENCH_OFFICIAL_STANDARD_COUNT = 159
HARMBENCH_OFFICIAL_TEST_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_test.csv"
)
DEFAULT_HARMBENCH_METADATA_PATH = (
    ROOT_DIR
    / "outputs"
    / "benchmark_cache"
    / "audit_sources"
    / "harmbench_behaviors_text_test.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score saved RIGJ StrongREJECT generations and/or recompute "
            "HarmBench standard-only LLM ASR from saved judgements."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("strongreject", "harmbench-standard", "all"),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Defaults to <run-dir>/{RESULTS_DIR_NAME}.",
    )
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=None,
        help="Judgement JSONL file/directory; defaults to <run-dir>/llm_asr_judge.",
    )
    parser.add_argument(
        "--harmbench-metadata",
        type=Path,
        default=None,
        help=(
            "Official HarmBench text_test CSV/JSONL carrying FunctionalCategory. "
            "Defaults to outputs/benchmark_cache/audit_sources/"
            "harmbench_behaviors_text_test.csv and must contain exactly 159 standard rows."
        ),
    )
    parser.add_argument(
        "--strongreject-rubric", choices=tuple(RUBRICS), default="maintained"
    )
    parser.add_argument(
        "--model-roles", nargs="+", choices=("source", "target"), default=None
    )
    parser.add_argument("--job-ids", nargs="+", default=None)
    parser.add_argument("--pair-ids", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--strongreject-deferred-retry-rounds", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)

    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-concurrency", type=int, default=None)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-parse-retries", type=int, default=2)
    parser.add_argument("--judge-api-retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--judge-no-api-retry-forever",
        action="store_false",
        dest="judge_api_retry_forever",
        default=True,
    )
    parser.add_argument("--judge-max-output-tokens", type=int, default=1024)
    return parser


def _load_dotenv(*, override: bool = False, path: Path | None = None) -> bool:
    path = path or (ROOT_DIR / ".env")
    if not path.is_file():
        return False
    changed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and (override or key not in os.environ):
            if os.environ.get(key) != value:
                changed = True
            os.environ[key] = value
    return changed


def _results_output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir or (Path(args.run_dir) / RESULTS_DIR_NAME))


def _judge_model_name(args: argparse.Namespace) -> str:
    return str(args.judge_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini")


def _judge_concurrency(args: argparse.Namespace) -> int:
    raw = (
        args.judge_concurrency
        if args.judge_concurrency is not None
        else os.getenv("LLM_ASR_JUDGE_CONCURRENCY") or "1"
    )
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"judge concurrency must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"judge concurrency must be >= 1, got {value}")
    return value


def _supports_temperature(model: str) -> bool:
    return not (
        str(model).lower().startswith("gpt-5")
        or "lucen.cc" in str(os.getenv("OPENAI_BASE_URL") or "").lower()
    )


def _make_client(args: argparse.Namespace) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or export it.")
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(args.judge_timeout_seconds),
        "max_retries": int(args.judge_max_retries),
    }
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(**kwargs)


def _runtime_signature(args: argparse.Namespace) -> tuple[str, str, str]:
    return (
        str(os.getenv("OPENAI_API_KEY") or ""),
        str(os.getenv("OPENAI_BASE_URL") or ""),
        _judge_model_name(args),
    )


class _ReloadableClient:
    def __init__(self, args: argparse.Namespace, client: OpenAI | None = None) -> None:
        self.args = args
        self._lock = threading.Lock()
        self._generation = 0
        self._signature = _runtime_signature(args)
        self._client = client or _make_client(args)
        self._model = _judge_model_name(args)

    def snapshot(self) -> tuple[OpenAI, str, int]:
        with self._lock:
            return self._client, self._model, self._generation

    def wait_for_credentials(
        self, *, failed_generation: int, sleep_seconds: float
    ) -> None:
        with self._lock:
            while self._generation == failed_generation:
                _load_dotenv(override=True)
                signature = _runtime_signature(self.args)
                changed = signature != self._signature
                print(
                    "[strongreject-retry] billing error; reloaded .env; "
                    f"credentials_changed={str(changed).lower()} sleep={sleep_seconds}s",
                    flush=True,
                )
                if changed:
                    self._client = _make_client(self.args)
                    self._model = _judge_model_name(self.args)
                    self._signature = signature
                    self._generation += 1
                    return
                time.sleep(sleep_seconds)


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif getattr(item, "text", None):
                    parts.append(str(item.text))
            return "\n".join(parts).strip()
    return str(getattr(response, "output_text", "") or "").strip()


def _is_balance_error(error: str) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "insufficient balance",
            "insufficient_balance",
            "insufficient quota",
            "insufficient_quota",
            "billing_error",
        )
    )


def _is_transient_error(error: str) -> bool:
    text = str(error).lower()
    if any(
        marker in text
        for marker in (
            "content_filter",
            "safety filter",
            "permission denied",
            "error code: 403",
        )
    ):
        return False
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "rate limit",
            "rate_limit",
            "connection",
            "bad gateway",
            "service unavailable",
            "internal server",
            "temporarily unavailable",
            "error code: 408",
            "error code: 429",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
        )
    )


def _invoke_strongreject(
    *,
    client: _ReloadableClient,
    rubric: StrongRejectRubric,
    prompt: str,
    response: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if rubric.system_prompt:
        messages.append({"role": "system", "content": rubric.system_prompt})
    messages.append({"role": "user", "content": rubric.render(prompt, response)})
    api_attempts = 0
    parse_attempts = 0
    last_raw = ""
    while True:
        api_attempts += 1
        api_client, model, generation = client.snapshot()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": int(args.judge_max_output_tokens),
            }
            if _supports_temperature(model):
                kwargs["temperature"] = float(args.judge_temperature)
            last_raw = _response_text(api_client.chat.completions.create(**kwargs))
            try:
                parsed = parse_strongreject_judgement(last_raw)
            except ValueError:
                parse_attempts += 1
                if parse_attempts <= int(args.judge_parse_retries):
                    continue
                raise
            return {
                **parsed.to_dict(),
                "grader_raw_output": last_raw,
                "grader_error": "",
                "grader_attempts": api_attempts,
                "grader_parse_attempts": parse_attempts + 1,
            }
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if args.judge_api_retry_forever and _is_balance_error(error):
                client.wait_for_credentials(
                    failed_generation=generation,
                    sleep_seconds=float(args.judge_api_retry_sleep_seconds),
                )
                continue
            if args.judge_api_retry_forever and _is_transient_error(error):
                print(
                    f"[strongreject-retry] transient error: {error[:500]}; "
                    f"sleep={float(args.judge_api_retry_sleep_seconds)}s",
                    flush=True,
                )
                time.sleep(float(args.judge_api_retry_sleep_seconds))
                continue
            raise


def _score_cache_key(
    *, rubric: StrongRejectRubric, model: str, prompt: str, response: str
) -> str:
    return stable_hash(
        {
            "rubric_fingerprint": rubric.fingerprint,
            "model": model,
            "prompt": prompt,
            "response": response,
        },
        length=64,
    )


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        str(row["cache_key"]): row for row in read_jsonl(path) if row.get("cache_key")
    }


def _resolve_resume_cache(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = [
        path / "strongreject" / "strongreject_grader_cache.jsonl",
        path / "strongreject_grader_cache.jsonl",
    ]
    matches = [candidate for candidate in candidates if candidate.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one strongreject_grader_cache.jsonl under {path}, got {matches}"
        )
    return matches[0]


def _selected_strongreject_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_generation_rows(
        args.run_dir,
        benchmark="strongreject",
        model_roles=set(args.model_roles) if args.model_roles else None,
        job_ids=set(args.job_ids) if args.job_ids else None,
    )
    if args.pair_ids:
        pair_ids = set(map(str, args.pair_ids))
        source_families = {
            pair_id.split("_to_", 1)[0] for pair_id in pair_ids if "_to_" in pair_id
        }
        rows = [
            row
            for row in rows
            if str(row.get("pair_id")) in pair_ids
            or (
                row.get("model_role") == "source"
                and str(row.get("attack_source_family")) in source_families
            )
        ]
    if args.max_samples is not None:
        if int(args.max_samples) < 1:
            raise ValueError("--max-samples must be >= 1")
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise RuntimeError(
            f"no saved StrongREJECT generations found under {args.run_dir}"
        )
    return rows


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    return float(statistics.fmean(items)) if items else None


def _strongreject_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("strongreject_score") is not None]
    return {
        "n": len(rows),
        "scored_count": len(scored),
        "error_count": len(rows) - len(scored),
        "mean_score": _mean(row["strongreject_score"] for row in scored),
        "median_score": (
            float(statistics.median(row["strongreject_score"] for row in scored))
            if scored
            else None
        ),
        "refusal_rate": _mean(row["strongreject_refusal"] for row in scored),
        "mean_convincingness": _mean(
            row["strongreject_convincingness"] for row in scored
        ),
        "mean_specificity": _mean(row["strongreject_specificity"] for row in scored),
    }


def _strongreject_summary(
    rows: list[dict[str, Any]],
    *,
    manifests: dict[str, dict[str, Any]],
    rubric: StrongRejectRubric,
    model: str,
) -> dict[str, Any]:
    by_job: dict[str, Any] = {}
    job_ids = sorted({str(row.get("job_id")) for row in rows})
    for job_id in job_ids:
        items = [row for row in rows if str(row.get("job_id")) == job_id]
        meta = items[0] if items else manifests.get(job_id, {})
        by_job[job_id] = {
            "attack_source_family": meta.get("attack_source_family"),
            "eval_model_family": meta.get("eval_model_family")
            or meta.get("eval_family"),
            "model_role": meta.get("model_role"),
            **_strongreject_group(items),
        }
    target_pairs = sorted(
        {
            str(row.get("pair_id"))
            for row in rows
            if row.get("model_role") == "target" and "_to_" in str(row.get("pair_id"))
        }
    )
    by_pair: dict[str, Any] = {}
    for pair_id in target_pairs:
        source_family = pair_id.split("_to_", 1)[0]
        source = _strongreject_group(
            [
                row
                for row in rows
                if row.get("model_role") == "source"
                and str(row.get("attack_source_family")) == source_family
            ]
        )
        target = _strongreject_group(
            [row for row in rows if str(row.get("pair_id")) == pair_id]
        )
        source_score = source["mean_score"]
        target_score = target["mean_score"]
        by_pair[pair_id] = {
            "source": source,
            "target": target,
            "score_gap": (
                target_score - source_score
                if source_score is not None and target_score is not None
                else None
            ),
        }
    return {
        "report_version": STRONGREJECT_REPORT_VERSION,
        "created_at": utc_timestamp(),
        "evaluator": "strongreject_rubric",
        "rubric_version": rubric.name,
        "rubric_source_commit": rubric.source_commit,
        "rubric_fingerprint": rubric.fingerprint,
        "judge_model": model,
        "replication_fidelity": (
            "rubric_and_model_exact"
            if rubric.name == "paper_2024" and model == "gpt-4-1106-preview"
            else (
                "rubric_exact_model_substituted"
                if rubric.name == "paper_2024"
                else "maintained_rubric_configured_model"
            )
        ),
        "overall": _strongreject_group(rows),
        "by_benchmark": {"strongreject": _strongreject_group(rows)},
        "by_generation_job": by_job,
        "by_pair": by_pair,
    }


def _metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _strongreject_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# StrongREJECT Saved-Output Re-score",
        "",
        f"- rubric: `{summary['rubric_version']}`",
        f"- judge: `{summary['judge_model']}`",
        f"- fidelity: `{summary['replication_fidelity']}`",
        "",
        "## Overall",
        "",
        "| n | scored | mean | median | refusal | convincingness | specificity | errors |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {overall['n']} | {overall['scored_count']} | "
            f"{_metric(overall['mean_score'])} | {_metric(overall['median_score'])} | "
            f"{_metric(overall['refusal_rate'])} | "
            f"{_metric(overall['mean_convincingness'])} | "
            f"{_metric(overall['mean_specificity'])} | {overall['error_count']} |"
        ),
        "",
        "## By generation job",
        "",
        "| job | source | eval | role | scored | mean | refusal | errors |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for job_id, value in summary["by_generation_job"].items():
        lines.append(
            f"| `{job_id}` | {value.get('attack_source_family')} | "
            f"{value.get('eval_model_family')} | {value.get('model_role')} | "
            f"{value['scored_count']}/{value['n']} | {_metric(value['mean_score'])} | "
            f"{_metric(value['refusal_rate'])} | {value['error_count']} |"
        )
    lines.extend(
        [
            "",
            "## By model pair",
            "",
            "| pair | source scored | source score | target scored | target score | gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair_id, value in summary["by_pair"].items():
        source = value["source"]
        target = value["target"]
        lines.append(
            f"| {pair_id} | {source['scored_count']}/{source['n']} | "
            f"{_metric(source['mean_score'])} | {target['scored_count']}/{target['n']} | "
            f"{_metric(target['mean_score'])} | {_metric(value['score_gap'])} |"
        )
    return "\n".join(lines) + "\n"


def _print_strongreject_summary(summary: dict[str, Any]) -> None:
    print("[strongreject-final-summary]")
    print("pair\tsource\ttarget\tgap\terrors")
    for pair_id, value in summary.get("by_pair", {}).items():
        source = value["source"]
        target = value["target"]
        print(
            f"{pair_id}\t{_metric(source['mean_score'])}\t"
            f"{_metric(target['mean_score'])}\t{_metric(value['score_gap'])}\t"
            f"{source['error_count']}/{target['error_count']}"
        )


def run_strongreject(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _selected_strongreject_rows(args)
    manifests = load_job_manifests(args.run_dir)
    rubric = get_strongreject_rubric(args.strongreject_rubric)
    model = _judge_model_name(args)
    existing_by_key: dict[str, dict[str, Any]] = {}
    judge_dir = Path(args.judge_dir or (Path(args.run_dir) / "llm_asr_judge"))
    if judge_dir.exists():
        existing_rows, _ = load_judgements(judge_dir)
        existing_by_key = {judgement_key(row): row for row in existing_rows}

    input_rows: list[dict[str, Any]] = []
    key_by_sample: dict[str, str] = {}
    unique_inputs: dict[str, dict[str, Any]] = {}
    for sample in samples:
        cache_key = _score_cache_key(
            rubric=rubric,
            model=model,
            prompt=row_prompt(sample),
            response=str(sample.get("model_output") or ""),
        )
        sample_key = judgement_key(sample)
        key_by_sample[sample_key] = cache_key
        unique_inputs.setdefault(cache_key, sample)
        input_rows.append(
            {
                **sample,
                "cache_key": cache_key,
                "rubric_version": rubric.name,
                "rubric_fingerprint": rubric.fingerprint,
                "judge_model": model,
            }
        )
    write_jsonl(output_dir / "strongreject_inputs.jsonl", input_rows)

    cache_path = output_dir / "strongreject_grader_cache.jsonl"
    if args.restart and args.resume:
        raise ValueError("--restart and --resume are mutually exclusive")
    if args.resume_from is not None and not args.resume:
        raise ValueError("--resume-from requires --resume")
    if (
        cache_path.exists()
        and not args.resume
        and not args.restart
        and not args.dry_run
    ):
        raise FileExistsError(f"{cache_path} exists; use --resume or --restart")
    if args.restart and cache_path.exists() and not args.dry_run:
        cache_path.unlink()
    resume_path = (
        _resolve_resume_cache(args.resume_from)
        if args.resume and args.resume_from is not None
        else cache_path
    )
    cache = _load_cache(resume_path) if args.resume else {}
    if args.resume and resume_path != cache_path:
        write_jsonl(cache_path, cache.values())
    pending = [
        (key, sample)
        for key, sample in unique_inputs.items()
        if key not in cache or cache[key].get("score") is None
    ]
    print(
        f"[strongreject] samples={len(samples)} unique={len(unique_inputs)} "
        f"cached={len(unique_inputs) - len(pending)} pending={len(pending)} "
        f"rubric={rubric.name} model={model}",
        flush=True,
    )
    if args.dry_run:
        summary = {
            "report_version": STRONGREJECT_REPORT_VERSION,
            "dry_run": True,
            "samples": len(samples),
            "unique_calls": len(unique_inputs),
            "cached_calls": len(unique_inputs) - len(pending),
            "pending_calls": len(pending),
            "job_count": len({str(row.get("job_id")) for row in samples}),
            "pair_count": len(
                {
                    str(row.get("pair_id"))
                    for row in samples
                    if row.get("model_role") == "target"
                }
            ),
            "rubric_version": rubric.name,
            "rubric_source_commit": rubric.source_commit,
            "rubric_fingerprint": rubric.fingerprint,
            "judge_model": model,
            "judge_concurrency": _judge_concurrency(args),
        }
        write_json(output_dir / "strongreject_dry_run_summary.json", summary)
        return summary

    retry_rounds = int(args.strongreject_deferred_retry_rounds)
    if retry_rounds < 0:
        raise ValueError("--strongreject-deferred-retry-rounds must be >= 0")
    reloadable = _ReloadableClient(args) if pending else None
    concurrency = _judge_concurrency(args)

    def evaluate(
        item: tuple[str, dict[str, Any]], *, phase: str, retry_round: int
    ) -> tuple[str, dict[str, Any]]:
        cache_key, sample = item
        try:
            if reloadable is None:
                raise RuntimeError("missing judge client")
            result = _invoke_strongreject(
                client=reloadable,
                rubric=rubric,
                prompt=row_prompt(sample),
                response=str(sample.get("model_output") or ""),
                args=args,
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "refusal": None,
                "convincingness": None,
                "specificity": None,
                "score": None,
                "grader_raw_output": "",
                "grader_error": f"{type(exc).__name__}: {exc}",
                "grader_attempts": None,
                "grader_parse_attempts": None,
            }
        return cache_key, {
            "cache_key": cache_key,
            "rubric_version": rubric.name,
            "rubric_fingerprint": rubric.fingerprint,
            "judge_model": model,
            "grader_phase": phase,
            "grader_retry_round": retry_round,
            **result,
        }

    def scoring_pass(
        items: list[tuple[str, dict[str, Any]]], *, phase: str, retry_round: int
    ) -> list[tuple[str, dict[str, Any]]]:
        failed: list[tuple[str, dict[str, Any]]] = []
        completed = 0

        def record(
            item: tuple[str, dict[str, Any]], key: str, result: dict[str, Any]
        ) -> None:
            cache[key] = result
            append_jsonl(cache_path, result)
            if result.get("score") is None:
                failed.append(item)

        def progress() -> None:
            if args.print_every > 0 and completed % int(args.print_every) == 0:
                print(f"[strongreject-{phase}] {completed}/{len(items)}", flush=True)

        if concurrency == 1:
            for item in items:
                key, result = evaluate(item, phase=phase, retry_round=retry_round)
                record(item, key, result)
                completed += 1
                progress()
            return failed
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            iterator = iter(items)
            active: dict[Any, tuple[str, dict[str, Any]]] = {}
            for _ in range(min(concurrency, len(items))):
                item = next(iterator, None)
                if item is not None:
                    active[
                        executor.submit(
                            evaluate, item, phase=phase, retry_round=retry_round
                        )
                    ] = item
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    item = active.pop(future)
                    key, result = future.result()
                    record(item, key, result)
                    completed += 1
                    progress()
                    next_item = next(iterator, None)
                    if next_item is not None:
                        active[
                            executor.submit(
                                evaluate,
                                next_item,
                                phase=phase,
                                retry_round=retry_round,
                            )
                        ] = next_item
        return failed

    deferred = scoring_pass(pending, phase="initial", retry_round=0)
    initial_failures = len(deferred)
    for retry_round in range(1, retry_rounds + 1):
        if not deferred:
            break
        print(
            f"[strongreject-deferred-retry] round={retry_round} pending={len(deferred)}",
            flush=True,
        )
        deferred = scoring_pass(
            deferred, phase="deferred_retry", retry_round=retry_round
        )

    scored_rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_key = judgement_key(sample)
        cache_row = cache[key_by_sample[sample_key]]
        score = cache_row.get("score")
        if score is not None:
            expected = score_strongreject(
                cache_row["refusal"],
                cache_row["convincingness"],
                cache_row["specificity"],
            )
            if abs(float(score) - expected) > 1e-12:
                raise RuntimeError(
                    f"cached StrongREJECT formula mismatch for {sample_key}"
                )
        row = {
            **sample,
            "strongreject_rubric_version": rubric.name,
            "strongreject_judge_model": model,
            "strongreject_refusal": cache_row.get("refusal"),
            "strongreject_convincingness": cache_row.get("convincingness"),
            "strongreject_specificity": cache_row.get("specificity"),
            "strongreject_score": score,
            "strongreject_grader_raw_output": cache_row.get("grader_raw_output", ""),
            "strongreject_grader_error": cache_row.get("grader_error", ""),
            "strongreject_grader_attempts": cache_row.get("grader_attempts"),
            "strongreject_grader_phase": cache_row.get("grader_phase"),
            "strongreject_grader_retry_round": cache_row.get("grader_retry_round"),
        }
        existing = existing_by_key.get(sample_key)
        if existing is not None:
            row.update(
                {
                    "existing_any_harmful_output": existing.get("any_harmful_output"),
                    "existing_prompt_matched_attack_success": existing.get(
                        "prompt_matched_attack_success"
                    ),
                    "existing_judge_error": existing.get("judge_error"),
                }
            )
        scored_rows.append(row)
    summary = _strongreject_summary(
        scored_rows, manifests=manifests, rubric=rubric, model=model
    )
    summary.update(
        {
            "judge_concurrency": concurrency,
            "initial_deferred_error_count": initial_failures,
            "recovered_after_deferred_retry_count": initial_failures - len(deferred),
            "unresolved_after_deferred_retry_count": len(deferred),
            "deferred_retry_rounds": retry_rounds,
            "existing_llm_asr_joined_count": sum(
                int("existing_prompt_matched_attack_success" in row)
                for row in scored_rows
            ),
        }
    )
    write_jsonl(output_dir / "strongreject_scores.jsonl", scored_rows)
    write_csv(
        output_dir / "strongreject_scores.csv",
        scored_rows,
        preferred_fields=(
            "job_id",
            "pair_id",
            "model_role",
            "attack_source_family",
            "eval_model_family",
            "sample_id",
            "goal",
            "strongreject_refusal",
            "strongreject_convincingness",
            "strongreject_specificity",
            "strongreject_score",
            "strongreject_grader_error",
        ),
    )
    write_json(output_dir / "strongreject_summary.json", summary)
    (output_dir / "strongreject_summary.md").write_text(
        _strongreject_markdown(summary), encoding="utf-8"
    )
    _print_strongreject_summary(summary)
    return summary


def _metadata_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return read_jsonl(path)


def _harmbench_record(row: dict[str, Any]) -> dict[str, str]:
    prompt = row_prompt(row)
    sample_id = str(
        row.get("id")
        or row.get("BehaviorID")
        or row.get("behavior_id")
        or row.get("original_id")
        or ""
    )
    category = ""
    for key in (
        "FunctionalCategory",
        "functional_category",
        "harmbench_category",
        "category",
    ):
        candidate = str(row.get(key) or "").strip().lower()
        if candidate in _HARMBENCH_CATEGORIES:
            category = candidate
            break
    if (
        not sample_id
        or not normalize_prompt(prompt)
        or category not in _HARMBENCH_CATEGORIES
    ):
        raise ValueError(f"invalid HarmBench metadata row: {row}")
    return {"id": sample_id, "prompt": prompt, "category": category}


def _harmbench_indexes(
    args: argparse.Namespace,
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    dict[str, Any],
]:
    by_id: dict[str, list[dict[str, str]]] = {}
    by_prompt: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    path = Path(args.harmbench_metadata or DEFAULT_HARMBENCH_METADATA_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"official HarmBench text_test metadata not found: {path}; "
            "download the official CSV or pass --harmbench-metadata"
        )
    raw_rows = _metadata_rows(path)
    for raw in raw_rows:
        record = _harmbench_record(raw)
        key = (record["id"], normalize_prompt(record["prompt"]), record["category"])
        if key in seen:
            raise ValueError(f"duplicate HarmBench metadata row: {key}")
        if record["id"] in by_id:
            raise ValueError(f"duplicate HarmBench BehaviorID: {record['id']}")
        seen.add(key)
        by_id.setdefault(record["id"], []).append(record)
        by_prompt.setdefault(normalize_prompt(record["prompt"]), []).append(record)
    category_counts = Counter(
        record["category"] for values in by_id.values() for record in values
    )
    if category_counts["standard"] != HARMBENCH_OFFICIAL_STANDARD_COUNT:
        raise ValueError(
            "official HarmBench text_test metadata must contain exactly "
            f"{HARMBENCH_OFFICIAL_STANDARD_COUNT} standard rows, got "
            f"{category_counts['standard']} from {path}"
        )
    metadata = {
        "official_url": HARMBENCH_OFFICIAL_TEST_URL,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(seen),
        "category_counts": {
            category: int(category_counts[category])
            for category in sorted(_HARMBENCH_CATEGORIES)
        },
    }
    return by_id, by_prompt, metadata


def _unique_metadata(items: list[dict[str, str]]) -> dict[str, str] | None:
    unique = {
        (item["id"], normalize_prompt(item["prompt"]), item["category"]): item
        for item in items
    }
    categories = {key[2] for key in unique}
    return next(iter(unique.values())) if unique and len(categories) == 1 else None


def _attach_harmbench_category(
    row: dict[str, Any],
    *,
    by_id: dict[str, list[dict[str, str]]],
    by_prompt: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    prompt = normalize_prompt(row_prompt(row))
    candidate_ids = [
        ("original_id", str(row.get("original_id") or "")),
        ("sample_id", str(row.get("sample_id") or "")),
    ]
    for id_field, candidate_id in candidate_ids:
        if not candidate_id:
            continue
        id_candidates = by_id.get(candidate_id, [])
        prompt_matches = [
            item for item in id_candidates if normalize_prompt(item["prompt"]) == prompt
        ]
        match = _unique_metadata(prompt_matches)
        if match is None:
            if id_candidates and not prompt_matches:
                raise ValueError(
                    f"HarmBench prompt mismatch for {candidate_id}: "
                    f"judge={prompt!r} metadata has "
                    f"{sorted({normalize_prompt(item['prompt']) for item in id_candidates})!r}"
                )
            continue
        return {
            **row,
            "harmbench_category": match["category"],
            "harmbench_metadata_id": match["id"],
            "harmbench_match_method": id_field,
        }
    prompt_candidates = by_prompt.get(prompt, [])
    match = _unique_metadata(prompt_candidates)
    if match is None and prompt_candidates:
        raise ValueError(
            f"HarmBench row {row.get('sample_id')!r} has ambiguous prompt metadata"
        )
    if match is None:
        return {
            **row,
            "harmbench_category": _HARMBENCH_NOT_IN_OFFICIAL_TEST,
            "harmbench_metadata_id": None,
            "harmbench_match_method": _HARMBENCH_NOT_IN_OFFICIAL_TEST,
        }
    return {
        **row,
        "harmbench_category": match["category"],
        "harmbench_metadata_id": match["id"],
        "harmbench_match_method": "prompt_exact",
    }


def _resolve_judge_dir(args: argparse.Namespace) -> Path:
    path = Path(args.judge_dir or (Path(args.run_dir) / "llm_asr_judge"))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _harmbench_standard_markdown(summary: dict[str, Any]) -> str:
    metadata = summary["harmbench_metadata"]
    category_counts = summary["category_counts"]
    lines = [
        "# HarmBench Standard-Only Common LLM ASR",
        "",
        "## Official text_test filter audit",
        "",
        f"- metadata: `{metadata['path']}`",
        f"- official URL: `{metadata['official_url']}`",
        f"- SHA-256: `{metadata['sha256']}`",
        (
            f"- metadata rows: {metadata['row_count']} "
            f"(standard={metadata['category_counts']['standard']}, "
            f"contextual={metadata['category_counts']['contextual']}, "
            f"copyright={metadata['category_counts']['copyright']})"
        ),
        "",
        "| input HarmBench rows | official standard matched | contextual excluded | "
        "copyright excluded | not in official test excluded |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['original_harmbench_row_count']} | "
            f"{summary['official_standard_matched_row_count']} | "
            f"{category_counts['contextual']} | {category_counts['copyright']} | "
            f"{category_counts[_HARMBENCH_NOT_IN_OFFICIAL_TEST]} |"
        ),
        "",
        "### Match methods",
        "",
        "| method | rows |",
        "| --- | ---: |",
    ]
    for method, count in summary["match_method_counts"].items():
        lines.append(f"| {method} | {count} |")
    base_lines = judgement_summary_markdown(
        summary,
        title="HarmBench Standard-Only Common LLM ASR",
    ).splitlines()
    lines.extend(["", *base_lines[2:], ""])
    return "\n".join(lines)


def run_harmbench_standard(
    args: argparse.Namespace, output_dir: Path
) -> dict[str, Any]:
    judge_dir = _resolve_judge_dir(args)
    judgements, judge_paths = load_judgements(judge_dir)
    rows = [
        row
        for row in judgements
        if str(row.get("benchmark") or "").lower() == "harmbench"
    ]
    model_roles = set(getattr(args, "model_roles", None) or [])
    job_ids = set(getattr(args, "job_ids", None) or [])
    if model_roles:
        rows = [row for row in rows if str(row.get("model_role")) in model_roles]
    if job_ids:
        rows = [row for row in rows if str(row.get("job_id")) in job_ids]
    pair_ids = set(map(str, getattr(args, "pair_ids", None) or []))
    if pair_ids:
        source_families = {
            pair_id.split("_to_", 1)[0] for pair_id in pair_ids if "_to_" in pair_id
        }
        rows = [
            row
            for row in rows
            if str(row.get("pair_id")) in pair_ids
            or (
                row.get("model_role") == "source"
                and str(row.get("attack_source_family")) in source_families
            )
        ]
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None:
        if int(max_samples) < 1:
            raise ValueError("--max-samples must be >= 1")
        rows = rows[: int(max_samples)]
    if not rows:
        raise RuntimeError(f"no HarmBench rows found under {judge_dir}")
    by_id, by_prompt, metadata = _harmbench_indexes(args)
    annotated = [
        _attach_harmbench_category(row, by_id=by_id, by_prompt=by_prompt)
        for row in rows
    ]
    standard_rows = [
        row for row in annotated if row["harmbench_category"] == "standard"
    ]
    original_summary = summarize_judgements(annotated)
    summary = summarize_judgements(standard_rows)
    category_counts = {
        category: sum(row["harmbench_category"] == category for row in annotated)
        for category in sorted(
            {*_HARMBENCH_CATEGORIES, _HARMBENCH_NOT_IN_OFFICIAL_TEST}
        )
    }
    match_method_counts = Counter(row["harmbench_match_method"] for row in annotated)
    excluded_nonstandard_count = len(annotated) - len(standard_rows)
    summary.update(
        {
            "report_version": HARMBENCH_STANDARD_REPORT_VERSION,
            "metric_name": "HarmBench-standard common LLM ASR",
            "original_overall": original_summary["overall"],
            "category_counts": category_counts,
            "excluded_contextual_count": category_counts["contextual"],
            "excluded_copyright_count": category_counts["copyright"],
            "excluded_not_in_official_test_count": category_counts[
                _HARMBENCH_NOT_IN_OFFICIAL_TEST
            ],
            "excluded_nonstandard_count": excluded_nonstandard_count,
            "original_harmbench_row_count": len(annotated),
            "official_standard_matched_row_count": len(standard_rows),
            "match_method_counts": dict(sorted(match_method_counts.items())),
            "judge_paths": [str(path.resolve()) for path in judge_paths],
            "harmbench_metadata": metadata,
            "harmbench_metadata_paths": [metadata["path"]],
        }
    )
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary["dry_run"] = True
        write_json(output_dir / "harmbench_standard_dry_run_summary.json", summary)
        return summary
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "harmbench_standard_rows.jsonl", standard_rows)
    write_csv(
        output_dir / "harmbench_standard_rows.csv",
        standard_rows,
        preferred_fields=(
            "judge_sample_key",
            "job_id",
            "pair_id",
            "model_role",
            "sample_id",
            "goal",
            "harmbench_category",
            "harmbench_match_method",
            "any_harmful_output",
            "prompt_matched_attack_success",
            "judge_error",
        ),
    )
    write_json(output_dir / "harmbench_standard_summary.json", summary)
    (output_dir / "harmbench_standard_summary.md").write_text(
        _harmbench_standard_markdown(summary),
        encoding="utf-8",
    )
    print("[harmbench-standard-final-summary]")
    print(
        "pair\tsource_judged/standard\tsource_asr\t"
        "target_judged/standard\ttarget_asr\tgap"
    )
    for pair_id, value in summary["by_pair"].items():
        source = value["source"]
        target = value["target"]
        print(
            f"{pair_id}\t{source['judged_count']}/{source['selected_count']}\t"
            f"{source['asr']}\t{target['judged_count']}/{target['selected_count']}\t"
            f"{target['asr']}\t{value['asr_gap']}"
        )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    _load_dotenv()
    if not Path(args.run_dir).exists():
        raise FileNotFoundError(args.run_dir)
    results_dir = _results_output_dir(args)
    print(f"[results-dir] {results_dir}")
    if args.mode in {"strongreject", "all"}:
        run_strongreject(args, results_dir / "strongreject")
    if args.mode in {"harmbench-standard", "all"}:
        run_harmbench_standard(args, results_dir / "harmbench_standard")


if __name__ == "__main__":
    main()
