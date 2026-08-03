#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.benchmark_io import (  # noqa: E402
    MODEL_FAMILY_ORDER,
    PAPER_HARMFUL_BENCHMARKS,
    atomic_write_json,
    directed_model_pairs,
    load_model_config,
    utc_timestamp,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all 20 directed source->target model pairs sequentially."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--attack-bank",
        type=Path,
        required=True,
        help=(
            "attack_assets directory produced by prepare_attack_assets.py, or "
            "its parent run directory."
        ),
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional subset such as qwen:llama mistral:vicuna.",
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attack-position",
        choices=["prefix", "suffix"],
        default="prefix",
    )
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-source-model-mismatch", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the 20-pair plan and commands without loading models.",
    )
    return parser


def _resolve_attack_bank(path: Path) -> Path:
    path = path.resolve()
    if (path / "attack_bank_manifest.json").is_file():
        return path
    if (path / "attack_assets" / "attack_bank_manifest.json").is_file():
        return path / "attack_assets"
    raise FileNotFoundError(
        f"cannot find attack_bank_manifest.json in {path} or {path / 'attack_assets'}"
    )


def _parse_pairs(values: list[str] | None) -> list[tuple[str, str]]:
    all_pairs = directed_model_pairs(MODEL_FAMILY_ORDER)
    if values is None:
        return all_pairs
    selected: list[tuple[str, str]] = []
    for value in values:
        parts = value.lower().split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid pair {value!r}; expected source:target")
        pair = (parts[0], parts[1])
        if pair not in all_pairs:
            raise ValueError(f"unsupported directed pair: {value}")
        if pair not in selected:
            selected.append(pair)
    return selected


def _pair_command(
    *,
    args: argparse.Namespace,
    models: dict[str, dict],
    attack_bank: Path,
    source: str,
    target: str,
) -> list[str]:
    pair_dir = args.output_dir / "pairs" / f"{source}_to_{target}"
    command = [
        sys.executable,
        "-u",
        str(ROOT_DIR / "scripts" / "run_harm_benchmark_eval.py"),
        "--output-dir",
        str(pair_dir),
        "--source-model",
        models[source]["path"],
        "--target-model",
        models[target]["path"],
        "--source-family",
        source,
        "--target-family",
        target,
        "--attack-artifact",
        str(attack_bank / source / "attack.json"),
        "--attack-position",
        args.attack_position,
        "--benchmarks",
        *args.benchmarks,
        "--benchmark-cache-dir",
        str(args.benchmark_cache_dir),
        "--benchmark-max-examples",
        str(args.benchmark_max_examples),
        "--benchmark-timeout-seconds",
        str(args.benchmark_timeout_seconds),
        "--seed",
        str(args.seed),
        "--gen-batch-size",
        str(args.gen_batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device-map",
        args.device_map,
        "--torch-dtype",
        args.torch_dtype,
        "--generation-cache-dir",
        str(args.output_dir / "generations"),
    ]
    if args.max_input_tokens is not None:
        command.extend(["--max-input-tokens", str(args.max_input_tokens)])
    if (
        args.trust_remote_code
        or bool(models[source].get("trust_remote_code"))
        or bool(models[target].get("trust_remote_code"))
    ):
        command.append("--trust-remote-code")
    if args.local_files_only:
        command.append("--local-files-only")
    if args.allow_source_model_mismatch:
        command.append("--allow-source-model-mismatch")
    return command


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model_config(args.model_config)
    attack_bank = _resolve_attack_bank(args.attack_bank)
    pairs = _parse_pairs(args.pairs)
    if args.pairs is None and len(pairs) != 20:
        raise RuntimeError(f"expected 20 directed pairs, got {len(pairs)}")

    manifest = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "model_config": str(args.model_config.resolve()),
        "attack_bank": str(attack_bank),
        "planned_pairs": [f"{source}_to_{target}" for source, target in pairs],
        "completed_pairs": [],
        "generation_cache_dir": str((args.output_dir / "generations").resolve()),
        "dry_run": bool(args.dry_run),
        "commands": [],
    }
    manifest_path = args.output_dir / "all_pairs_manifest.json"
    atomic_write_json(manifest_path, manifest)

    for index, (source, target) in enumerate(pairs, start=1):
        pair_id = f"{source}_to_{target}"
        command = _pair_command(
            args=args,
            models=models,
            attack_bank=attack_bank,
            source=source,
            target=target,
        )
        print(
            f"[pair {index}/{len(pairs)}] {pair_id}: {' '.join(command)}",
            flush=True,
        )
        manifest["commands"].append({"pair_id": pair_id, "command": command})
        atomic_write_json(manifest_path, manifest)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT_DIR, check=True)
        manifest["completed_pairs"].append(
            {
                "pair_id": pair_id,
                "completed_at": utc_timestamp(),
                "manifest": str(
                    (
                        args.output_dir / "pairs" / pair_id / "pair_manifest.json"
                    ).resolve()
                ),
            }
        )
        atomic_write_json(manifest_path, manifest)

    manifest["completed_at"] = utc_timestamp()
    atomic_write_json(manifest_path, manifest)
    label = "planned" if args.dry_run else "completed"
    print(f"[done] {label} {len(pairs)} pairs: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
