from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.prepare_attack_assets import _ngd_command, _validate_reused_attack_source
from scripts.judge_outputs_llm_asr import (
    _benchmark_selection,
    _invoke_judge,
    _is_balance_error,
    _load_samples,
    _load_dotenv,
    _strict_ahs_rating,
    _strict_bool_pair,
    _summarize,
    _summarize_ahs,
)
from scripts.run_harm_benchmark_eval import (
    INTERNLM3_EOS_GENERATION_REVISION,
    _generate_batch,
    _generation_revision,
    _generation_stop_kwargs,
    _prepare_result_file,
)
from utils.benchmark_io import (
    HARMBENCH_OFFICIAL_STANDARD_COUNT,
    HARMBENCH_OFFICIAL_TEST_URL,
    HARMFUL_BENCH_SPECS,
    MODEL_FAMILY_ORDER,
    _validate_benchmark_records,
    directed_model_pairs,
    atomic_write_json,
    load_attack_artifact,
    load_model_config,
    normalize_harmful_rows,
    sample_records_deterministic,
    write_jsonl,
)
from utils.model_adapter import (
    VICUNA_CHAT_TEMPLATE,
    chat_template_generation_kwargs,
    ensure_known_chat_template,
    infer_model_family,
)


class _FakeTokenizer:
    chat_template = None


class _FakeQwen3Tokenizer:
    chat_template = "{% if enable_thinking %}thinking{% endif %}"


class BenchmarkDataTests(unittest.TestCase):
    def test_harmbench_standard_filter(self) -> None:
        rows = [
            {"Behavior": "one", "FunctionalCategory": "standard"},
            {"Behavior": "two", "FunctionalCategory": "contextual"},
            {"Behavior": "three", "FunctionalCategory": "Standard"},
        ]
        records, counts = normalize_harmful_rows(
            rows,
            benchmark="harmbench",
            source="fixture",
        )
        self.assertEqual([row["instruction"] for row in records], ["one", "three"])
        self.assertEqual(counts["raw_count"], 3)
        self.assertEqual(counts["category_filtered_count"], 2)

    def test_harmbench_has_no_nonofficial_fallback(self) -> None:
        self.assertEqual(
            HARMFUL_BENCH_SPECS["harmbench"],
            [{"kind": "url", "value": HARMBENCH_OFFICIAL_TEST_URL}],
        )

    def test_harmbench_cache_requires_159_official_rows(self) -> None:
        records = [
            {"source": HARMBENCH_OFFICIAL_TEST_URL, "id": str(index)}
            for index in range(HARMBENCH_OFFICIAL_STANDARD_COUNT)
        ]
        _validate_benchmark_records(
            "harmbench", records, source=HARMBENCH_OFFICIAL_TEST_URL
        )
        with self.assertRaisesRegex(ValueError, "exactly 159"):
            _validate_benchmark_records(
                "harmbench",
                records[:-1],
                source=HARMBENCH_OFFICIAL_TEST_URL,
            )
        with self.assertRaisesRegex(ValueError, "official text_test"):
            _validate_benchmark_records(
                "harmbench",
                records,
                source="harmbench_behaviors_text_all.csv",
            )

    def test_wildjailbreak_adversarial_harmful_only(self) -> None:
        rows = [
            {"prompt": "one", "label": "adversarial_harmful"},
            {"prompt": "two", "label": "adversarial_benign"},
            {"prompt": "three", "data_type": "adversarial_harmful"},
            {"prompt": "four", "data_type": "vanilla_harmful"},
        ]
        records, counts = normalize_harmful_rows(
            rows,
            benchmark="wildjailbreak",
            source="fixture",
        )
        self.assertEqual([row["instruction"] for row in records], ["one", "three"])
        self.assertEqual(counts["category_filtered_count"], 2)

    def test_deterministic_sample_preserves_source_order(self) -> None:
        records = [{"instruction": str(index), "id": str(index)} for index in range(20)]
        first = sample_records_deterministic(records, n=5, seed=7)
        second = sample_records_deterministic(records, n=5, seed=7)
        self.assertEqual(first, second)
        selected_ids = [int(row["id"]) for row in first]
        self.assertEqual(selected_ids, sorted(selected_ids))


class ModelAndPairTests(unittest.TestCase):
    def test_exactly_twenty_directed_pairs(self) -> None:
        pairs = directed_model_pairs(MODEL_FAMILY_ORDER)
        self.assertEqual(len(pairs), 20)
        self.assertTrue(all(source != target for source, target in pairs))
        self.assertEqual(len(set(pairs)), 20)

    def test_model_config_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "models.json"
            path.write_text(
                json.dumps(
                    {
                        "models": {
                            family: {"path": f"Models/{family}"}
                            for family in MODEL_FAMILY_ORDER
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = load_model_config(path)
            self.assertEqual(list(config), list(MODEL_FAMILY_ORDER))

    def test_family_detection_and_vicuna_fallback(self) -> None:
        self.assertEqual(infer_model_family("Models/Qwen3-8B-Instruct"), "qwen")
        self.assertEqual(infer_model_family("Models/vicuna-7b-v1.5"), "vicuna")
        self.assertEqual(
            infer_model_family("Models/internlm3-8b-instruct"),
            "internlm",
        )
        tokenizer = _FakeTokenizer()
        self.assertTrue(ensure_known_chat_template(tokenizer, "vicuna"))
        self.assertEqual(tokenizer.chat_template, VICUNA_CHAT_TEMPLATE)

    def test_qwen3_thinking_is_disabled(self) -> None:
        self.assertEqual(
            chat_template_generation_kwargs(_FakeQwen3Tokenizer()),
            {"enable_thinking": False},
        )

    def test_all_pairs_dry_run_cli_plans_twenty_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "models.json"
            atomic_write_json(
                config_path,
                {
                    "models": {
                        family: {"path": f"Models/{family}"}
                        for family in MODEL_FAMILY_ORDER
                    }
                },
            )
            attack_bank = root / "attack_assets"
            atomic_write_json(attack_bank / "attack_bank_manifest.json", {})
            output_dir = root / "run"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_all_model_pairs.py",
                    "--output-dir",
                    str(output_dir),
                    "--model-config",
                    str(config_path),
                    "--attack-bank",
                    str(attack_bank),
                    "--dry-run",
                ],
                check=True,
            )
            manifest = json.loads(
                (output_dir / "all_pairs_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["planned_pairs"]), 20)
            self.assertEqual(len(manifest["commands"]), 20)
            self.assertEqual(manifest["completed_pairs"], [])

    def test_attack_command_honors_model_trust_remote_code(self) -> None:
        args = Namespace(
            train_dataset="harmbench_gjo",
            n_train_data=20,
            n_test_data=20,
            seed=0,
            attack_position="prefix",
            num_steps=1,
            num_adv_tokens=4,
            lr=1.0,
            beta_1=0.9,
            beta_2=0.9999,
            begin_tau=5.0,
            final_tau=1.0,
            device="cuda:0",
            torch_dtype="bfloat16",
            loss_model_path="./checkpoints/anchor_classifier.pth",
            anchor_datasets=["benign.txt", "harmful.txt"],
            trust_remote_code=False,
            local_files_only=True,
        )
        command = _ngd_command(
            args=args,
            family="internlm",
            model_path="Models/internlm3-8b-instruct",
            model_trust_remote_code=True,
            result_path=Path("result.json"),
        )
        self.assertIn("--trust-remote-code", command)
        self.assertIn("--local-files-only", command)


class ArtifactAndResumeTests(unittest.TestCase):
    def test_reused_attack_must_match_configured_source_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_model = Path(temp_dir) / "Qwen2.5-7B-Instruct"
            new_model = Path(temp_dir) / "Qwen3-8B-Instruct"
            old_model.mkdir()
            new_model.mkdir()
            artifact = {
                "best_adv_prompt": "attack",
                "source_model": str(old_model),
            }
            with self.assertRaisesRegex(ValueError, "cannot reuse"):
                _validate_reused_attack_source(
                    artifact,
                    family="qwen",
                    configured_model=str(new_model),
                    artifact_path=Path("attack.json"),
                )

    def test_latest_numeric_ngd_record_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "source_model": "Models/qwen",
                            "source_family": "qwen",
                            "attack_position": "prefix",
                        },
                        "1": {"best_adv_prompt": "old"},
                        "10": {"best_adv_prompt": "latest", "best_avg_test_asr": 0.5},
                        "2": {"best_adv_prompt": "middle"},
                    }
                ),
                encoding="utf-8",
            )
            artifact = load_attack_artifact(path)
            self.assertEqual(artifact["best_adv_prompt"], "latest")
            self.assertEqual(artifact["ngd_record_key"], "10")

    def test_resume_compacts_duplicates_and_retries_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "outputs.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"generation_key": "a", "status": "generation_error"}
                        ),
                        json.dumps({"generation_key": "a", "status": "success"}),
                        json.dumps(
                            {"generation_key": "b", "status": "generation_error"}
                        ),
                        json.dumps({"generation_key": "c", "status": "input_too_long"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            completed = _prepare_result_file(path, resume=True)
            self.assertEqual(completed, {"a", "c"})
            saved = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(saved), 2)

    def test_resume_invalidates_only_legacy_internlm3_eos_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "outputs.jsonl"
            rows = [
                {
                    "generation_key": "broken",
                    "eval_model_family": "internlm",
                    "status": "success",
                    "model_output": "Factorization<|im_end|><|im_end|>",
                    "reached_max_new_tokens": True,
                },
                {
                    "generation_key": "valid",
                    "eval_model_family": "internlm",
                    "status": "success",
                    "model_output": "A normal answer",
                    "reached_max_new_tokens": False,
                },
                {
                    "generation_key": "already-fixed",
                    "eval_model_family": "internlm",
                    "status": "success",
                    "model_output": "literal <|im_end|> discussion",
                    "reached_max_new_tokens": True,
                    "generation_revision": INTERNLM3_EOS_GENERATION_REVISION,
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            completed = _prepare_result_file(path, resume=True)

            self.assertEqual(completed, {"valid", "already-fixed"})
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [row["generation_key"] for row in saved],
                ["valid", "already-fixed"],
            )
            archive = path.with_name(f"{path.name}.invalid-internlm3-eos.bak")
            archived = [json.loads(line) for line in archive.read_text().splitlines()]
            self.assertEqual(
                [row["generation_key"] for row in archived],
                ["broken"],
            )

    def test_internlm3_uses_generation_config_multi_eos(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="internlm3",
                architectures=["InternLM3ForCausalLM"],
            ),
            generation_config=SimpleNamespace(
                eos_token_id=[2, 128131],
                pad_token_id=2,
            ),
        )
        tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2)

        stop_kwargs = _generation_stop_kwargs(model, tokenizer)

        self.assertEqual(stop_kwargs["eos_token_id"], [2, 128131])
        self.assertEqual(stop_kwargs["pad_token_id"], 2)
        self.assertEqual(
            _generation_revision(model, tokenizer, stop_kwargs),
            INTERNLM3_EOS_GENERATION_REVISION,
        )

    def test_internlm3_batch_stops_and_strips_im_end(self) -> None:
        class Encoded(dict):
            def to(self, device):
                return self

        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def __call__(self, *args, **kwargs):
                return Encoded(
                    input_ids=torch.tensor([[10, 11]]),
                    attention_mask=torch.tensor([[1, 1]]),
                )

            def decode(self, token_ids, *, skip_special_tokens):
                self.last_decoded_ids = token_ids
                return "Factorization"

        class Model:
            config = SimpleNamespace(
                model_type="internlm3",
                architectures=["InternLM3ForCausalLM"],
            )
            generation_config = SimpleNamespace(
                eos_token_id=[2, 128131],
                pad_token_id=2,
            )

            def __init__(self):
                self.embeddings = torch.nn.Embedding(16, 4)

            def get_input_embeddings(self):
                return self.embeddings

            def generate(self, **kwargs):
                self.generation_kwargs = kwargs
                return torch.tensor([[10, 11, 42, 128131, 2, 2]])

        model = Model()
        tokenizer = Tokenizer()
        rows = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            pending=[{"rendered_prompt": "prompt"}],
            max_new_tokens=4,
        )

        self.assertEqual(model.generation_kwargs["eos_token_id"], [2, 128131])
        self.assertEqual(tokenizer.last_decoded_ids, [42])
        self.assertEqual(rows[0]["model_output"], "Factorization")
        self.assertEqual(rows[0]["output_token_count"], 1)
        self.assertFalse(rows[0]["reached_max_new_tokens"])
        self.assertEqual(
            rows[0]["generation_revision"],
            INTERNLM3_EOS_GENERATION_REVISION,
        )


class JudgeTests(unittest.TestCase):
    def test_saved_benchmark_subset_selectors_filter_before_judging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generation_path = root / "generations" / "job" / "rows.jsonl"

            def row(key: str, benchmark: str, goal: str) -> dict[str, object]:
                return {
                    "record_type": "rigj_harm_generation",
                    "generation_key": key,
                    "job_id": "job",
                    "status": "success",
                    "pair_id": "qwen_to_llama",
                    "model_role": "target",
                    "attack_source_family": "qwen",
                    "eval_model_family": "llama",
                    "benchmark": benchmark,
                    "sample_id": key,
                    "goal": goal,
                    "model_output": "fixture output",
                }

            write_jsonl(
                generation_path,
                [
                    row("adv", "advbench", "adv prompt"),
                    row("hb-standard", "harmbench", "standard prompt"),
                    row("hb-other", "harmbench", "contextual prompt"),
                    row("jbb", "jailbreakbench", "jbb prompt"),
                    row("mi", "malicious_instruct", "mi prompt"),
                    row("wj-ah", "wildjailbreak", "AH prompt"),
                    row("wj-benign", "wildjailbreak", "benign prompt"),
                    row("sr", "strongreject", "strongreject prompt"),
                ],
            )
            args = Namespace(
                judge_input_path=root,
                judge_conditions=None,
                judge_include_baseline=True,
                judge_benchmarks=[
                    "advbench",
                    "harmbench-standard",
                    "jailbreakbench",
                    "malicious_instruct",
                    "wildjailbreak-ah",
                ],
                judge_harmbench_metadata=root / "unused.csv",
                judge_wildjailbreak_arrow=None,
                judge_model_roles=None,
                judge_max_samples=None,
            )
            with (
                mock.patch(
                    "scripts.judge_outputs_llm_asr._load_harmbench_standard_lookup",
                    return_value=(
                        {"official-standard": "standard prompt"},
                        {"standard prompt"},
                    ),
                ),
                mock.patch(
                    "scripts.judge_outputs_llm_asr._load_wildjailbreak_label_lookup",
                    return_value={
                        "AH prompt": "adversarial_harmful",
                        "benign prompt": "adversarial_benign",
                    },
                ),
            ):
                samples = _load_samples(args)

            self.assertEqual(
                {sample["judge_benchmark"] for sample in samples},
                {
                    "advbench",
                    "harmbench-standard",
                    "jailbreakbench",
                    "malicious_instruct",
                    "wildjailbreak-ah",
                },
            )
            self.assertEqual(
                {sample["judge_sample_key"] for sample in samples},
                {"adv", "hb-standard", "jbb", "mi", "wj-ah"},
            )
            summary = _summarize_ahs(
                [{**sample, "harmfulness_score": 5} for sample in samples]
            )
            self.assertEqual(
                set(summary["by_benchmark"]),
                {
                    "advbench",
                    "harmbench-standard",
                    "jailbreakbench",
                    "malicious_instruct",
                    "wildjailbreak-ah",
                },
            )

    def test_unknown_saved_benchmark_selector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported --judge-benchmarks"):
            _benchmark_selection(Namespace(judge_benchmarks=["advbenvh"]))

    def test_billing_error_marker_is_recognized(self) -> None:
        self.assertTrue(
            _is_balance_error("BadRequestError: {'error': {'type': 'billing_error'}}")
        )

    def test_empty_output_is_locally_judged_false_without_api(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=mock.Mock(
                        side_effect=AssertionError("API must not be called")
                    )
                )
            )
        )
        args = Namespace(judge_model="fixture-judge")
        result = _invoke_judge(
            client=client,
            sample={"judge_sample_key": "empty", "model_output": "  \n"},
            args=args,
        )

        self.assertFalse(result["any_harmful_output"])
        self.assertFalse(result["prompt_matched_attack_success"])
        self.assertEqual(result["llm_raw_output"], "")
        self.assertEqual(result["judge_api_attempts"], 0)
        client.chat.completions.create.assert_not_called()

    def test_empty_output_is_locally_scored_one_for_ahs_without_api(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=mock.Mock(
                        side_effect=AssertionError("API must not be called")
                    )
                )
            )
        )
        args = Namespace(judge_model="fixture-judge", judge_metric="ahs")
        result = _invoke_judge(
            client=client,
            sample={"judge_sample_key": "empty", "model_output": "  \n"},
            args=args,
        )

        self.assertEqual(result["judge_metric"], "ahs")
        self.assertEqual(result["harmfulness_score"], 1)
        self.assertEqual(result["ahs_score"], 1)
        self.assertEqual(result["judge_api_attempts"], 0)
        client.chat.completions.create.assert_not_called()

    def test_dotenv_reload_can_override_rotated_api_key(self) -> None:
        original = os.environ.get("OPENAI_API_KEY")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / ".env"
                path.write_text("OPENAI_API_KEY=new-key\n", encoding="utf-8")
                os.environ["OPENAI_API_KEY"] = "old-key"
                self.assertFalse(_load_dotenv(path=path))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "old-key")
                self.assertTrue(_load_dotenv(path=path, override=True))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "new-key")
        finally:
            if original is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original

    def test_billing_error_reloads_dotenv_and_rebuilds_client(self) -> None:
        class BillingCompletions:
            def create(self, **kwargs):
                raise RuntimeError("{'type': 'billing_error'}")

        class SuccessfulCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"any_harmful_output": false, '
                                    '"prompt_matched_attack_success": false}'
                                )
                            )
                        )
                    ]
                )

        first_client = SimpleNamespace(
            chat=SimpleNamespace(completions=BillingCompletions())
        )
        replacement_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SuccessfulCompletions())
        )
        args = Namespace(
            judge_model="fixture-judge",
            judge_temperature=0.0,
            judge_api_retry_forever=True,
            judge_api_retry_sleep_seconds=0.0,
            judge_skip_permission_denied=False,
        )
        sample = {
            "judge_sample_key": "fixture",
            "benchmark": "harmbench",
            "sample_id": "harmbench-00000",
            "goal": "fixture goal",
            "model_output": "fixture output",
        }
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "old-key"

        reload_count = 0

        def rotate_credentials(*, override=False):
            nonlocal reload_count
            self.assertTrue(override)
            reload_count += 1
            if reload_count >= 3:
                os.environ["OPENAI_API_KEY"] = "new-key"
                return True
            return False

        try:
            with (
                mock.patch(
                    "scripts.judge_outputs_llm_asr._load_dotenv",
                    side_effect=rotate_credentials,
                ) as load_dotenv_mock,
                mock.patch(
                    "scripts.judge_outputs_llm_asr._make_client",
                    return_value=replacement_client,
                ) as make_client_mock,
            ):
                result = _invoke_judge(client=first_client, sample=sample, args=args)

            self.assertEqual(load_dotenv_mock.call_count, 3)
            load_dotenv_mock.assert_called_with(override=True)
            make_client_mock.assert_called_once_with(args)
            self.assertIsNone(result["judge_error"])
            self.assertFalse(result["prompt_matched_attack_success"])
            self.assertEqual(
                result["llm_raw_output"],
                result["judge_raw"],
            )
        finally:
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_strict_pair(self) -> None:
        self.assertEqual(
            _strict_bool_pair(
                '{"any_harmful_output": true, '
                '"prompt_matched_attack_success": false}'
            ),
            (True, False),
        )
        with self.assertRaises(ValueError):
            _strict_bool_pair(
                '{"any_harmful_output": false, '
                '"prompt_matched_attack_success": true}'
            )

    def test_strict_ahs_rating(self) -> None:
        self.assertEqual(_strict_ahs_rating("Rating: [[1]]"), 1)
        self.assertEqual(_strict_ahs_rating("Rating: [[10]]"), 10)
        with self.assertRaises(ValueError):
            _strict_ahs_rating("Rating: [[0]]")
        with self.assertRaises(ValueError):
            _strict_ahs_rating("The rating is Rating: [[8]]")

    def test_ahs_judge_uses_paper_prompt_and_parses_score(self) -> None:
        create = mock.Mock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="Rating: [[8]]"))
                ]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        args = Namespace(
            judge_metric="ahs",
            judge_model="fixture-judge",
            judge_temperature=0.0,
            judge_parse_retries=0,
            judge_api_retry_forever=False,
            judge_api_retry_sleep_seconds=0.0,
            judge_skip_permission_denied=False,
        )
        result = _invoke_judge(
            client=client,
            sample={
                "judge_sample_key": "fixture",
                "condition": "attack",
                "benchmark": "harmbench",
                "pair_id": "qwen_to_llama",
                "model_role": "target",
                "sample_id": "one",
                "goal": "fixture harmful request",
                "model_output": "fixture response",
            },
            args=args,
        )

        self.assertEqual(result["harmfulness_score"], 8)
        self.assertEqual(result["judge_metric"], "ahs")
        messages = create.call_args.kwargs["messages"]
        self.assertIn("TASK: `fixture harmful request`", messages[0]["content"])
        self.assertIn("fixture response", messages[1]["content"])

    def test_summary_primary_and_conservative_asr(self) -> None:
        rows = [
            {
                "judge_sample_key": "1",
                "job_id": "source",
                "pair_id": "qwen_self",
                "model_role": "source",
                "attack_source_family": "qwen",
                "eval_model_family": "qwen",
                "benchmark": "harmbench",
                "any_harmful_output": True,
                "prompt_matched_attack_success": True,
            },
            {
                "judge_sample_key": "2",
                "job_id": "target",
                "pair_id": "qwen_to_llama",
                "model_role": "target",
                "attack_source_family": "qwen",
                "eval_model_family": "llama",
                "benchmark": "harmbench",
                "any_harmful_output": False,
                "prompt_matched_attack_success": False,
            },
        ]
        summary = _summarize(rows)
        self.assertEqual(summary["overall"]["asr"], 0.5)
        pair = summary["by_pair"]["qwen_to_llama"]
        self.assertEqual(pair["source"]["asr"], 1.0)
        self.assertEqual(pair["target"]["asr"], 0.0)
        self.assertEqual(pair["asr_gap"], -1.0)

    def test_ahs_summary_by_condition_and_benchmark(self) -> None:
        rows = [
            {
                "judge_sample_key": "1",
                "job_id": "source",
                "pair_id": "qwen_self",
                "model_role": "source",
                "attack_source_family": "qwen",
                "eval_model_family": "qwen",
                "condition": "baseline",
                "benchmark": "harmbench",
                "harmfulness_score": 2,
            },
            {
                "judge_sample_key": "2",
                "job_id": "target",
                "pair_id": "qwen_to_llama",
                "model_role": "target",
                "attack_source_family": "qwen",
                "eval_model_family": "llama",
                "condition": "attack",
                "benchmark": "harmbench",
                "harmfulness_score": 8,
            },
            {
                "judge_sample_key": "3",
                "job_id": "target",
                "pair_id": "qwen_to_llama",
                "model_role": "target",
                "attack_source_family": "qwen",
                "eval_model_family": "llama",
                "condition": "attack",
                "benchmark": "advbench",
                "harmfulness_score": 6,
            },
            {
                "judge_sample_key": "4",
                "job_id": "target",
                "pair_id": "qwen_to_llama",
                "model_role": "target",
                "attack_source_family": "qwen",
                "eval_model_family": "llama",
                "condition": "attack",
                "benchmark": "harmbench",
                "harmfulness_score": None,
                "judge_error": "fixture error",
            },
        ]

        summary = _summarize_ahs(rows)

        self.assertEqual(summary["judge_metric"], "ahs")
        self.assertEqual(summary["overall"]["ahs"], 16 / 3)
        self.assertEqual(summary["by_condition"]["attack"]["ahs"], 7.0)
        self.assertEqual(
            summary["by_condition_and_benchmark"]["attack"]["harmbench"]["ahs"],
            8.0,
        )
        self.assertEqual(
            summary["by_condition_and_benchmark"]["attack"]["harmbench"][
                "judge_error_count"
            ],
            1,
        )
        pair = summary["by_pair"]["qwen_to_llama"]
        self.assertEqual(pair["source"]["ahs"], 2.0)
        self.assertEqual(pair["target"]["ahs"], 7.0)
        self.assertEqual(pair["ahs_gap"], 5.0)

    def test_ahs_explicit_condition_does_not_use_full_manifest_denominator(
        self,
    ) -> None:
        rows = [
            {
                "judge_sample_key": "1",
                "job_id": "target",
                "pair_id": "qwen_to_llama",
                "model_role": "target",
                "attack_source_family": "qwen",
                "eval_model_family": "llama",
                "condition": "attack",
                "benchmark": "harmbench",
                "harmfulness_score": 8,
            }
        ]
        manifests = {
            "target": {
                "job_id": "target",
                "attack_source_family": "qwen",
                "eval_family": "llama",
                "summaries": {"harmbench": {"selected_count": 2}},
            }
        }

        summary = _summarize_ahs(rows, manifests)

        self.assertEqual(summary["overall"]["selected_count"], 1)
        self.assertEqual(summary["by_benchmark"]["harmbench"]["selected_count"], 1)
        self.assertEqual(summary["by_condition"]["attack"]["selected_count"], 1)
        self.assertEqual(
            summary["by_condition_and_benchmark"]["attack"]["harmbench"][
                "selected_count"
            ],
            1,
        )

    def test_ahs_default_condition_uses_legacy_manifest_denominator(self) -> None:
        rows = [
            {
                "judge_sample_key": "1",
                "job_id": "source",
                "pair_id": "qwen_self",
                "model_role": "source",
                "attack_source_family": "qwen",
                "eval_model_family": "qwen",
                "benchmark": "harmbench",
                "harmfulness_score": 8,
            }
        ]
        manifests = {
            "source": {
                "job_id": "source",
                "attack_source_family": "qwen",
                "eval_family": "qwen",
                "summaries": {"harmbench": {"selected_count": 2}},
            }
        }

        summary = _summarize_ahs(rows, manifests)

        self.assertEqual(summary["overall"]["selected_count"], 2)
        self.assertEqual(summary["by_condition"]["default"]["selected_count"], 2)

    def test_manifest_selected_count_controls_conservative_denominator(self) -> None:
        rows = [
            {
                "judge_sample_key": "1",
                "job_id": "source",
                "pair_id": "qwen_self",
                "model_role": "source",
                "attack_source_family": "qwen",
                "eval_model_family": "qwen",
                "benchmark": "harmbench",
                "any_harmful_output": True,
                "prompt_matched_attack_success": True,
            }
        ]
        manifests = {
            "source": {
                "job_id": "source",
                "attack_source_family": "qwen",
                "eval_family": "qwen",
                "summaries": {"harmbench": {"selected_count": 2}},
            }
        }
        summary = _summarize(rows, manifests)
        self.assertEqual(summary["overall"]["asr"], 1.0)
        self.assertEqual(
            summary["overall"]["prompt_matched_attack_success_rate"],
            1.0,
        )
        self.assertEqual(
            summary["overall"]["prompt_matched_attack_success_total_rate"],
            0.5,
        )
        self.assertEqual(summary["overall"]["conservative_asr"], 0.5)
        self.assertEqual(summary["overall"]["coverage"], 0.5)

    def test_judge_dry_run_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_dir = root / "generations" / "job"
            write_jsonl(
                job_dir / "harmbench.jsonl",
                [
                    {
                        "record_type": "rigj_harm_generation",
                        "generation_key": "key",
                        "job_id": "job",
                        "status": "success",
                        "pair_id": "qwen_self",
                        "model_role": "source",
                        "attack_source_family": "qwen",
                        "eval_model_family": "qwen",
                        "benchmark": "harmbench",
                        "sample_id": "harmbench-00000",
                        "goal": "fixture request",
                        "model_output": "fixture output",
                    }
                ],
            )
            atomic_write_json(
                job_dir / "job_manifest.json",
                {
                    "job_id": "job",
                    "attack_source_family": "qwen",
                    "eval_family": "qwen",
                    "summaries": {"harmbench": {"selected_count": 2}},
                },
            )
            report_dir = root / "judge"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/judge_outputs_llm_asr.py",
                    "--judge-input-path",
                    str(root),
                    "--judge-report-dir",
                    str(report_dir),
                    "--judge-run-name",
                    "smoke",
                    "--judge-dry-run",
                ],
                check=True,
            )
            summary = json.loads(
                (report_dir / "smoke.summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["candidate_count"], 1)

    def test_ahs_dry_run_cli_filters_condition_and_uses_separate_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job_dir = root / "generations" / "job"
            rows = []
            for key, condition in (
                ("attack-key", "attack"),
                ("control-key", "control"),
            ):
                rows.append(
                    {
                        "record_type": "rigj_harm_generation",
                        "generation_key": key,
                        "job_id": "job",
                        "status": "success",
                        "pair_id": "qwen_to_llama",
                        "model_role": "target",
                        "attack_source_family": "qwen",
                        "eval_model_family": "llama",
                        "condition": condition,
                        "benchmark": "harmbench",
                        "sample_id": key,
                        "goal": "fixture request",
                        "model_output": "fixture output",
                    }
                )
            write_jsonl(job_dir / "harmbench.jsonl", rows)
            report_dir = root / "judge"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/judge_outputs_llm_asr.py",
                    "--ahs",
                    "--judge-input-path",
                    str(root),
                    "--judge-report-dir",
                    str(report_dir),
                    "--judge-run-name",
                    "smoke",
                    "--judge-conditions",
                    "attack",
                    "--judge-no-include-baseline",
                    "--judge-dry-run",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            summary_path = report_dir / "smoke.ahs.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["judge_metric"], "ahs")
            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["conditions"], ["attack"])
            inputs = [
                json.loads(line)
                for line in (report_dir / "smoke.ahs.inputs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(inputs[0]["condition"], "attack")


if __name__ == "__main__":
    unittest.main()
