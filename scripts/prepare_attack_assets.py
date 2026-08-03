#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.benchmark_io import (  # noqa: E402
    MODEL_FAMILY_ORDER,
    PAPER_HARMFUL_BENCHMARKS,
    atomic_write_json,
    load_attack_artifact,
    load_harmful_benchmarks,
    load_model_config,
    materialize_attacked_prompts,
    stable_hash,
    utc_timestamp,
)
from utils.common import MODEL_NAME_TO_PATH  # noqa: E402
from utils.model_adapter import resolve_model_reference  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original RIGJ NGD attack once for each of the five source "
            "models and package reusable local attack artifacts."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-dataset", default="harmbench_gjo")
    parser.add_argument("--n-train-data", type=int, default=20)
    parser.add_argument("--n-test-data", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attack-position",
        choices=["prefix", "suffix"],
        default="prefix",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--num-adv-tokens", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--beta-1", type=float, default=0.9)
    parser.add_argument("--beta-2", type=float, default=0.9999)
    parser.add_argument("--begin-tau", type=float, default=5.0)
    parser.add_argument("--final-tau", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--loss-model-path",
        default="./checkpoints/anchor_classifier.pth",
    )
    parser.add_argument(
        "--anchor-datasets",
        nargs=2,
        default=[
            "./data/prompt-driven_benign.txt",
            "./data/prompt-driven_harmful.txt",
        ],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=MODEL_FAMILY_ORDER,
        default=list(MODEL_FAMILY_ORDER),
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--skip-benchmark-prompts", action="store_true")
    parser.add_argument(
        "--benchmark-cache-dir",
        type=Path,
        default=Path("outputs/benchmark_cache"),
    )
    parser.add_argument("--benchmark-max-examples", type=int, default=199)
    parser.add_argument("--benchmark-timeout-seconds", type=int, default=30)
    return parser


def _ngd_command(
    *,
    args: argparse.Namespace,
    family: str,
    model_path: str,
    model_trust_remote_code: bool,
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT_DIR / "ngd_main.py"),
        "--source-model",
        model_path,
        "--source-family",
        family,
        "--train-dataset",
        args.train_dataset,
        "--n-train-data",
        str(args.n_train_data),
        "--n-test-data",
        str(args.n_test_data),
        "--seed",
        str(args.seed),
        "--attack-position",
        args.attack_position,
        "--num-steps",
        str(args.num_steps),
        "--num-adv-tokens",
        str(args.num_adv_tokens),
        "--lr",
        str(args.lr),
        "--beta-1",
        str(args.beta_1),
        "--beta-2",
        str(args.beta_2),
        "--begin-tau",
        str(args.begin_tau),
        "--final-tau",
        str(args.final_tau),
        "--device",
        args.device,
        "--torch-dtype",
        args.torch_dtype,
        "--loss-model-path",
        args.loss_model_path,
        "--anchor_datasets",
        *args.anchor_datasets,
        "--save-folder",
        str(result_path),
    ]
    if args.trust_remote_code or model_trust_remote_code:
        command.append("--trust-remote-code")
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def _normalize_model_reference(reference: str) -> str:
    resolved, _ = resolve_model_reference(reference, MODEL_NAME_TO_PATH)
    candidate = Path(resolved)
    return str(candidate.resolve()) if candidate.exists() else resolved


def _validate_reused_attack_source(
    artifact: dict[str, Any],
    *,
    family: str,
    configured_model: str,
    artifact_path: Path,
) -> None:
    recorded_model = (
        artifact.get("model_load_metadata", {}).get("resolved_model_reference")
        or artifact.get("resolved_source_model")
        or artifact.get("source_model")
    )
    if not recorded_model:
        raise ValueError(
            f"cannot reuse {artifact_path}: attack artifact does not record its "
            "source model"
        )
    recorded = _normalize_model_reference(str(recorded_model))
    configured = _normalize_model_reference(configured_model)
    if recorded != configured:
        raise ValueError(
            f"cannot reuse {artifact_path}: {family} attack was optimized on "
            f"{recorded}, but the model config selects {configured}; use a new "
            "attack bank or rerun this family with --restart"
        )


def _package_attack(
    *,
    raw_result_path: Path,
    family: str,
    model: dict[str, Any],
    args: argparse.Namespace,
    command: list[str],
) -> dict[str, Any]:
    artifact = load_attack_artifact(raw_result_path)
    attack_id = stable_hash(
        {
            "family": family,
            "model": model["path"],
            "prompt": artifact["best_adv_prompt"],
            "position": args.attack_position,
            "train_dataset": args.train_dataset,
            "n_train_data": args.n_train_data,
            "seed": args.seed,
        },
        length=24,
    )
    return {
        **artifact,
        "schema_version": 1,
        "attack_id": attack_id,
        "source_family": family,
        "source_model": model["path"],
        "resolved_source_model": artifact.get("model_load_metadata", {}).get(
            "resolved_model_reference"
        ),
        "optimized_position": args.attack_position,
        "train_dataset": args.train_dataset,
        "n_train_data": args.n_train_data,
        "n_test_data": args.n_test_data,
        "seed": args.seed,
        "optimization_config": {
            "num_steps": args.num_steps,
            "num_adv_tokens": args.num_adv_tokens,
            "lr": args.lr,
            "beta_1": args.beta_1,
            "beta_2": args.beta_2,
            "begin_tau": args.begin_tau,
            "final_tau": args.final_tau,
            "internal_max_new_tokens": 24,
        },
        "command": command,
        "packaged_at": utc_timestamp(),
    }


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_model_config(args.model_config)
    attack_bank_dir = args.output_dir / "attack_assets"
    attack_bank_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, Any]] = {}
    for family in args.families:
        model = models[family]
        family_dir = attack_bank_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        raw_result_path = family_dir / "original_ngd_result.json"
        artifact_path = family_dir / "attack.json"
        if artifact_path.is_file() and not args.restart:
            artifact = load_attack_artifact(artifact_path)
            _validate_reused_attack_source(
                artifact,
                family=family,
                configured_model=model["path"],
                artifact_path=artifact_path,
            )
            print(f"[attack] reuse {family}: {artifact_path}", flush=True)
        else:
            command = _ngd_command(
                args=args,
                family=family,
                model_path=model["path"],
                model_trust_remote_code=bool(model.get("trust_remote_code")),
                result_path=raw_result_path,
            )
            atomic_write_json(
                family_dir / "command.json",
                {
                    "family": family,
                    "model": model,
                    "command": command,
                    "started_at": utc_timestamp(),
                },
            )
            print(f"[attack] start {family}: {' '.join(command)}", flush=True)
            env = dict(os.environ)
            env.setdefault("MPLBACKEND", "Agg")
            subprocess.run(
                command,
                cwd=ROOT_DIR,
                env=env,
                check=True,
            )
            artifact = _package_attack(
                raw_result_path=raw_result_path,
                family=family,
                model=model,
                args=args,
                command=command,
            )
            atomic_write_json(artifact_path, artifact)
            print(f"[attack] saved {family}: {artifact_path}", flush=True)
        artifacts[family] = artifact

    benchmark_manifest: dict[str, Any] | None = None
    if not args.skip_benchmark_prompts:
        records, benchmark_manifest = load_harmful_benchmarks(
            cache_dir=args.benchmark_cache_dir,
            output_dir=args.output_dir / "benchmark_selection",
            benchmark_names=PAPER_HARMFUL_BENCHMARKS,
            max_examples=args.benchmark_max_examples,
            seed=args.seed,
            timeout_seconds=args.benchmark_timeout_seconds,
        )
        for family, artifact in artifacts.items():
            materialize_attacked_prompts(
                attack_artifact=artifact,
                benchmark_records=records,
                output_dir=attack_bank_dir / family,
                attack_position=args.attack_position,
            )

    bank_manifest = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "model_config": str(args.model_config.resolve()),
        "families": {
            family: {
                "model": models[family],
                "attack_artifact": str(
                    (attack_bank_dir / family / "attack.json").resolve()
                ),
                "attack_id": artifact.get("attack_id"),
            }
            for family, artifact in artifacts.items()
        },
        "benchmark_selection_id": (
            benchmark_manifest.get("selection_id")
            if benchmark_manifest is not None
            else None
        ),
    }
    atomic_write_json(attack_bank_dir / "attack_bank_manifest.json", bank_manifest)
    print(
        f"[done] attack bank: {attack_bank_dir / 'attack_bank_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
