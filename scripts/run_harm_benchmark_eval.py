#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.benchmark_io import (  # noqa: E402
    PAPER_HARMFUL_BENCHMARKS,
    append_jsonl,
    atomic_write_json,
    build_attack_text,
    load_attack_artifact,
    load_harmful_benchmarks,
    read_jsonl,
    rule_based_jailbreak,
    stable_hash,
    utc_timestamp,
    write_jsonl,
)
from utils.common import MODEL_NAME_TO_PATH  # noqa: E402
from utils.model_adapter import (  # noqa: E402
    load_model_and_tokenizer,
    model_context_limit,
    model_input_device,
    render_user_prompt,
    resolve_model_reference,
)

INTERNLM3_EOS_GENERATION_REVISION = "internlm3_generation_config_eos_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one RIGJ source->target pair on the nine harmful benchmarks. "
            "Source and target are loaded sequentially and all per-sample outputs are saved."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model", default=None)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--source-family", default=None)
    parser.add_argument("--target-family", default=None)
    parser.add_argument("--attack-artifact", type=Path, default=None)
    parser.add_argument("--best-adv-prompt", default=None)
    parser.add_argument(
        "--attack-position",
        choices=["prefix", "suffix"],
        default="prefix",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(PAPER_HARMFUL_BENCHMARKS),
    )
    parser.add_argument(
        "--benchmark-cache-dir",
        type=Path,
        default=Path("outputs/benchmark_cache"),
    )
    parser.add_argument("--benchmark-max-examples", type=int, default=199)
    parser.add_argument("--benchmark-timeout-seconds", type=int, default=30)
    parser.add_argument("--benchmark-refresh", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help=(
            "Optional stricter input limit. Inputs beyond the effective model limit "
            "are recorded as input_too_long and are never silently truncated."
        ),
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--generation-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume", default=True
    )
    parser.add_argument("--allow-source-model-mismatch", action="store_true")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download, normalize, select, and save benchmark data without loading models.",
    )
    return parser


def _normalize_reference(reference: str) -> str:
    resolved, _ = resolve_model_reference(reference, MODEL_NAME_TO_PATH)
    candidate = Path(resolved)
    return str(candidate.resolve()) if candidate.exists() else resolved


def _validate_attack_source(
    artifact: dict[str, Any],
    source_model: str,
    *,
    allow_mismatch: bool,
) -> None:
    recorded = (
        artifact.get("model_load_metadata", {}).get("resolved_model_reference")
        or artifact.get("resolved_source_model")
        or artifact.get("source_model")
    )
    if not recorded:
        return
    expected = _normalize_reference(str(recorded))
    actual = _normalize_reference(source_model)
    if expected != actual and not allow_mismatch:
        raise ValueError(
            "attack artifact source model mismatch: "
            f"artifact={expected!r} requested={actual!r}; "
            "pass --allow-source-model-mismatch only for an intentional override"
        )


def _prepare_result_file(
    path: Path,
    *,
    resume: bool,
) -> set[str]:
    if not resume:
        write_jsonl(path, [])
        return set()
    rows = read_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        key = str(row.get("generation_key", ""))
        if not key:
            continue
        if key not in latest:
            order.append(key)
        latest[key] = row
    invalid_legacy = {
        key: row
        for key, row in latest.items()
        if _is_legacy_internlm3_eos_output(row)
    }
    terminal = {
        key: row
        for key, row in latest.items()
        if key not in invalid_legacy
        and row.get("status") in {"success", "input_too_long"}
    }
    kept = [terminal[key] for key in order if key in terminal]
    if invalid_legacy:
        archive_path = path.with_name(
            f"{path.name}.invalid-internlm3-eos.bak"
        )
        archived = {
            str(row.get("generation_key", "")): row
            for row in read_jsonl(archive_path)
            if row.get("generation_key")
        }
        archived.update(invalid_legacy)
        write_jsonl(archive_path, archived.values())
    write_jsonl(path, kept)
    if invalid_legacy:
        print(
            f"[resume] invalidated_legacy_internlm3_eos={len(invalid_legacy)} "
            f"path={path} archive={archive_path}",
            flush=True,
        )
    return set(terminal)


def _is_legacy_internlm3_eos_output(row: dict[str, Any]) -> bool:
    """Identify outputs produced before InternLM3's second EOS was honored."""
    return (
        str(row.get("eval_model_family") or "").lower() == "internlm"
        and row.get("status") == "success"
        and row.get("reached_max_new_tokens") is True
        and "<|im_end|>" in str(row.get("model_output") or "")
        and row.get("generation_revision")
        != INTERNLM3_EOS_GENERATION_REVISION
    )


def _is_internlm3_model(model: Any) -> bool:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    architectures = [
        str(name).lower()
        for name in (getattr(config, "architectures", None) or [])
    ]
    return model_type == "internlm3" or "internlm3forcausallm" in architectures


def _generation_stop_kwargs(model: Any, tokenizer: Any) -> dict[str, Any]:
    """Match COMBAT's InternLM3 generation-config stop-token handling."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if _is_internlm3_model(model):
        generation_config = getattr(model, "generation_config", None)
        configured_eos = getattr(generation_config, "eos_token_id", None)
        configured_pad = getattr(generation_config, "pad_token_id", None)
        if configured_eos is not None:
            eos_token_id = configured_eos
        if configured_pad is not None:
            pad_token_id = configured_pad

    kwargs: dict[str, Any] = {}
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    return kwargs


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {int(token_id) for token_id in values if token_id is not None}


def _generation_revision(
    model: Any,
    tokenizer: Any,
    stop_kwargs: dict[str, Any],
) -> str | None:
    if not _is_internlm3_model(model):
        return None
    resolved = _token_id_set(stop_kwargs.get("eos_token_id"))
    tokenizer_only = _token_id_set(getattr(tokenizer, "eos_token_id", None))
    if resolved != tokenizer_only:
        return INTERNLM3_EOS_GENERATION_REVISION
    return None


def _effective_input_limit(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    max_input_tokens: int | None,
) -> tuple[int | None, int | None]:
    context_limit = model_context_limit(model, tokenizer)
    effective = None
    if context_limit is not None:
        effective = max(0, context_limit - int(max_new_tokens))
    if max_input_tokens is not None:
        effective = (
            int(max_input_tokens)
            if effective is None
            else min(effective, int(max_input_tokens))
        )
    return context_limit, effective


def _base_generation_row(
    *,
    artifact: dict[str, Any],
    record: dict[str, Any],
    attack_position: str,
    attack_prompt: str,
    rendered_prompt: str,
    job_id: str,
    eval_model: str,
    eval_family: str,
    model_metadata: dict[str, Any],
    generation_config: dict[str, Any],
    input_token_count: int,
) -> dict[str, Any]:
    generation_key = stable_hash(
        {
            "job_id": job_id,
            "benchmark": record["benchmark"],
            "sample_id": record["id"],
            "generation_config": generation_config,
        },
        length=32,
    )
    attack_source_family = str(artifact.get("source_family") or "unknown")
    is_source = attack_source_family == eval_family
    return {
        "schema_version": 1,
        "record_type": "rigj_harm_generation",
        "generation_key": generation_key,
        "job_id": job_id,
        "attack_id": artifact.get("attack_id"),
        "attack_source_family": attack_source_family,
        "eval_model_family": eval_family,
        "model_role": "source" if is_source else "target",
        "pair_id": (
            f"{attack_source_family}_self"
            if is_source
            else f"{attack_source_family}_to_{eval_family}"
        ),
        "eval_model": eval_model,
        "model_load_metadata": model_metadata,
        "benchmark": record["benchmark"],
        "sample_id": record["id"],
        "original_id": record.get("original_id"),
        "category": record.get("category"),
        "dataset_source": record.get("source"),
        "goal": record["instruction"],
        "best_adv_prompt": artifact["best_adv_prompt"],
        "optimized_position": artifact.get("optimized_position", "prefix"),
        "attack_position": attack_position,
        "attack_prompt": attack_prompt,
        "rendered_prompt": rendered_prompt,
        "input_token_count": int(input_token_count),
        "generation_config": generation_config,
        "created_at": utc_timestamp(),
    }


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    pending: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rendered = [row["rendered_prompt"] for row in pending]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=False,
    ).to(model_input_device(model))
    input_width = int(encoded["input_ids"].shape[1])
    stop_kwargs = _generation_stop_kwargs(model, tokenizer)
    generation_revision = _generation_revision(model, tokenizer, stop_kwargs)
    started = time.monotonic()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            **stop_kwargs,
        )
    elapsed = time.monotonic() - started
    results: list[dict[str, Any]] = []
    pad_id = stop_kwargs.get("pad_token_id")
    eos_ids = _token_id_set(stop_kwargs.get("eos_token_id"))
    for index, row in enumerate(pending):
        output_ids = generated[index, input_width:].detach().cpu().tolist()
        while output_ids and pad_id is not None and output_ids[-1] == pad_id:
            output_ids.pop()
        reached_max_new_tokens = len(output_ids) >= int(max_new_tokens)
        while output_ids and output_ids[-1] in eos_ids:
            output_ids.pop()
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        result = {
            **row,
            "status": "success",
            "model_output": output,
            "output_token_count": len(output_ids),
            "reached_max_new_tokens": reached_max_new_tokens,
            "generation_seconds": elapsed / max(len(pending), 1),
            "rule_based_jailbreak": rule_based_jailbreak(output),
            "error": None,
        }
        if generation_revision is not None:
            result["generation_revision"] = generation_revision
            result["generation_stop_config"] = {
                "eos_token_id": stop_kwargs.get("eos_token_id"),
                "pad_token_id": pad_id,
            }
        results.append(result)
    return results


def _generate_with_individual_fallback(
    *,
    model: Any,
    tokenizer: Any,
    pending: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    try:
        return _generate_batch(
            model=model,
            tokenizer=tokenizer,
            pending=pending,
            max_new_tokens=max_new_tokens,
        )
    except Exception as batch_exc:  # noqa: BLE001
        results: list[dict[str, Any]] = []
        for row in pending:
            try:
                results.extend(
                    _generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        pending=[row],
                        max_new_tokens=max_new_tokens,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        **row,
                        "status": "generation_error",
                        "model_output": "",
                        "output_token_count": 0,
                        "reached_max_new_tokens": False,
                        "generation_seconds": None,
                        "rule_based_jailbreak": None,
                        "error": (
                            f"{type(exc).__name__}: {exc}; "
                            f"batch_error={type(batch_exc).__name__}: {batch_exc}"
                        ),
                    }
                )
        return results


def _summarize_file(path: Path, selected_count: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    latest = {
        str(row.get("generation_key")): row for row in rows if row.get("generation_key")
    }
    values = list(latest.values())
    success = [row for row in values if row.get("status") == "success"]
    rule_hits = sum(bool(row.get("rule_based_jailbreak")) for row in success)
    return {
        "selected_count": int(selected_count),
        "saved_count": len(values),
        "success_count": len(success),
        "input_too_long_count": sum(
            row.get("status") == "input_too_long" for row in values
        ),
        "generation_error_count": sum(
            row.get("status") == "generation_error" for row in values
        ),
        "coverage": len(success) / selected_count if selected_count else 0.0,
        "rule_based_asr": rule_hits / len(success) if success else None,
        "rule_based_conservative_asr": (
            rule_hits / selected_count if selected_count else None
        ),
    }


def generate_model_outputs(
    *,
    args: argparse.Namespace,
    artifact: dict[str, Any],
    benchmark_records: dict[str, list[dict[str, Any]]],
    eval_model: str,
    eval_family: str,
) -> dict[str, Any]:
    generation_config = {
        "do_sample": False,
        "max_new_tokens": int(args.max_new_tokens),
        "max_input_tokens": args.max_input_tokens,
        "torch_dtype": args.torch_dtype,
    }
    selection_fingerprint = stable_hash(
        {
            benchmark: [
                {
                    "id": record["id"],
                    "instruction": record["instruction"],
                }
                for record in records
            ]
            for benchmark, records in benchmark_records.items()
        }
    )
    job_id = stable_hash(
        {
            "attack_id": artifact["attack_id"],
            "eval_model": _normalize_reference(eval_model),
            "eval_family": eval_family,
            "attack_position": args.attack_position,
            "generation_config": generation_config,
            "selection_fingerprint": selection_fingerprint,
        },
        length=24,
    )
    cache_root = args.generation_cache_dir or (args.output_dir / "generations")
    job_dir = Path(cache_root) / (
        f"{artifact.get('source_family', 'attack')}_attack__on__{eval_family}_{job_id}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)

    all_terminal = True
    completed_by_benchmark: dict[str, set[str]] = {}
    for benchmark, records in benchmark_records.items():
        result_path = job_dir / f"{benchmark}.jsonl"
        completed = _prepare_result_file(result_path, resume=args.resume)
        completed_by_benchmark[benchmark] = completed
        if len(completed) < len(records):
            all_terminal = False

    model = tokenizer = None
    existing_manifest_path = job_dir / "job_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.is_file()
        else {}
    )
    model_metadata: dict[str, Any] = dict(
        existing_manifest.get("model_load_metadata") or {}
    )
    resolved_generation_stop_config: dict[str, Any] = dict(
        existing_manifest.get("generation_stop_config") or {}
    )
    resolved_generation_revision = existing_manifest.get("generation_revision")
    if not all_terminal:
        print(f"[model] loading {eval_family}: {eval_model}", flush=True)
        model, tokenizer, model_metadata = load_model_and_tokenizer(
            eval_model,
            aliases=MODEL_NAME_TO_PATH,
            family=eval_family,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            use_fast=False,
        )
        context_limit, input_limit = _effective_input_limit(
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )
        print(
            f"[model] context_limit={context_limit} effective_input_limit={input_limit}",
            flush=True,
        )
        stop_kwargs = _generation_stop_kwargs(model, tokenizer)
        resolved_generation_stop_config = {
            "eos_token_id": stop_kwargs.get("eos_token_id"),
            "pad_token_id": stop_kwargs.get("pad_token_id"),
        }
        resolved_generation_revision = _generation_revision(
            model,
            tokenizer,
            stop_kwargs,
        )
        print(
            "[model] generation_stop_config="
            f"{json.dumps(resolved_generation_stop_config, sort_keys=True)} "
            f"generation_revision={resolved_generation_revision}",
            flush=True,
        )

        try:
            for benchmark, records in benchmark_records.items():
                result_path = job_dir / f"{benchmark}.jsonl"
                completed = completed_by_benchmark[benchmark]
                batch: list[dict[str, Any]] = []
                for record in records:
                    attack_prompt = build_attack_text(
                        artifact["best_adv_prompt"],
                        str(record["instruction"]),
                        args.attack_position,
                    )
                    rendered = render_user_prompt(
                        tokenizer,
                        attack_prompt,
                        family=eval_family,
                        add_generation_prompt=True,
                    )
                    input_ids = tokenizer(
                        rendered,
                        add_special_tokens=False,
                        truncation=False,
                    ).input_ids
                    base_row = _base_generation_row(
                        artifact=artifact,
                        record=record,
                        attack_position=args.attack_position,
                        attack_prompt=attack_prompt,
                        rendered_prompt=rendered,
                        job_id=job_id,
                        eval_model=eval_model,
                        eval_family=eval_family,
                        model_metadata=model_metadata,
                        generation_config=generation_config,
                        input_token_count=len(input_ids),
                    )
                    if base_row["generation_key"] in completed:
                        continue
                    if input_limit is not None and len(input_ids) > input_limit:
                        append_jsonl(
                            result_path,
                            {
                                **base_row,
                                "status": "input_too_long",
                                "model_output": "",
                                "output_token_count": 0,
                                "reached_max_new_tokens": False,
                                "generation_seconds": None,
                                "rule_based_jailbreak": None,
                                "error": (
                                    f"input tokens {len(input_ids)} exceed effective "
                                    f"limit {input_limit}; no truncation was applied"
                                ),
                            },
                        )
                        continue
                    batch.append(base_row)
                    if len(batch) >= int(args.gen_batch_size):
                        for output_row in _generate_with_individual_fallback(
                            model=model,
                            tokenizer=tokenizer,
                            pending=batch,
                            max_new_tokens=args.max_new_tokens,
                        ):
                            append_jsonl(result_path, output_row)
                        batch = []
                if batch:
                    for output_row in _generate_with_individual_fallback(
                        model=model,
                        tokenizer=tokenizer,
                        pending=batch,
                        max_new_tokens=args.max_new_tokens,
                    ):
                        append_jsonl(result_path, output_row)
                print(
                    f"[generate] {eval_family}/{benchmark}: "
                    f"{_summarize_file(result_path, len(records))}",
                    flush=True,
                )
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summaries = {
        benchmark: _summarize_file(job_dir / f"{benchmark}.jsonl", len(records))
        for benchmark, records in benchmark_records.items()
    }
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "attack_id": artifact["attack_id"],
        "attack_source_family": artifact.get("source_family"),
        "eval_model": eval_model,
        "eval_family": eval_family,
        "attack_position": args.attack_position,
        "generation_config": generation_config,
        "generation_stop_config": resolved_generation_stop_config,
        "generation_revision": resolved_generation_revision,
        "selection_fingerprint": selection_fingerprint,
        "model_load_metadata": model_metadata,
        "result_files": {
            benchmark: str((job_dir / f"{benchmark}.jsonl").resolve())
            for benchmark in benchmark_records
        },
        "summaries": summaries,
        "updated_at": utc_timestamp(),
    }
    atomic_write_json(job_dir / "job_manifest.json", manifest)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_records, benchmark_manifest = load_harmful_benchmarks(
        cache_dir=args.benchmark_cache_dir,
        output_dir=args.output_dir,
        benchmark_names=args.benchmarks,
        max_examples=args.benchmark_max_examples,
        seed=args.seed,
        timeout_seconds=args.benchmark_timeout_seconds,
        refresh=args.benchmark_refresh,
    )
    if args.download_only:
        print(
            f"[done] downloaded selection={benchmark_manifest['selection_id']} "
            f"manifest={args.output_dir / 'benchmark_manifest.json'}",
            flush=True,
        )
        return
    missing = [
        name
        for name in (
            "source_model",
            "target_model",
            "source_family",
            "target_family",
            "attack_artifact",
        )
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise ValueError(
            f"non-download runs require: {', '.join('--' + name.replace('_', '-') for name in missing)}"
        )
    artifact = load_attack_artifact(args.attack_artifact)
    if args.best_adv_prompt is not None:
        artifact = {
            **artifact,
            "best_adv_prompt": args.best_adv_prompt,
            "attack_id": stable_hash(
                {
                    "base_attack_id": artifact.get("attack_id"),
                    "override": args.best_adv_prompt,
                }
            ),
        }
    _validate_attack_source(
        artifact,
        args.source_model,
        allow_mismatch=args.allow_source_model_mismatch,
    )
    run_config = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "args": vars(args),
        "attack_artifact": artifact,
        "benchmark_selection_id": benchmark_manifest["selection_id"],
    }
    atomic_write_json(args.output_dir / "run_config.json", run_config)

    source_job = generate_model_outputs(
        args=args,
        artifact=artifact,
        benchmark_records=benchmark_records,
        eval_model=args.source_model,
        eval_family=args.source_family,
    )
    target_job = generate_model_outputs(
        args=args,
        artifact=artifact,
        benchmark_records=benchmark_records,
        eval_model=args.target_model,
        eval_family=args.target_family,
    )
    pair_id = f"{args.source_family}_to_{args.target_family}"
    pair_manifest = {
        "schema_version": 1,
        "pair_id": pair_id,
        "source_family": args.source_family,
        "target_family": args.target_family,
        "source_model": args.source_model,
        "target_model": args.target_model,
        "attack_id": artifact["attack_id"],
        "benchmark_selection_id": benchmark_manifest["selection_id"],
        "source_job": source_job,
        "target_job": target_job,
        "completed_at": utc_timestamp(),
    }
    atomic_write_json(args.output_dir / "pair_manifest.json", pair_manifest)
    print(
        f"[done] pair={pair_id} manifest={args.output_dir / 'pair_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
