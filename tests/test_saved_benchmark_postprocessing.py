from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.audit_judge_outputs_benign_samples import (
    SourceRecord,
    ahs_paper_five_markdown,
    annotate_row,
    corrected_asr_rows,
    filter_ahs_paper_five_rows,
    run_audit,
)
from scripts.judge_outputs_llm_asr import _summarize, _summarize_ahs
from scripts.recompute_saved_benchmark_scores import (
    _harmbench_indexes,
    run_harmbench_standard,
    run_strongreject,
)
from utils.saved_benchmark_postprocess import (
    load_generation_rows,
    load_judgements,
    normalize_prompt,
    read_jsonl,
    summarize_judgements,
    write_jsonl,
)
from utils.strongreject import (
    RUBRICS,
    parse_strongreject_judgement,
    score_strongreject,
)

EXPECTED_MAINTAINED_FINGERPRINT = (
    "00d3053cebf0bd06648870de7ffde8679b82d950472566b04864ab74da9dc888"
)
EXPECTED_PAPER_FINGERPRINT = (
    "59c76e2d7ec35a5f83cbb315bf7edf59d2764a2eb8c4684e2f0be5c3a248fa53"
)


def _generation_row(
    *,
    key: str,
    benchmark: str = "strongreject",
    prompt: str = "test prompt",
    output: str = "test response",
    job_id: str = "source-job",
    pair_id: str = "qwen_self",
    role: str = "source",
    source: str = "qwen",
    target: str = "qwen",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "rigj_harm_generation",
        "generation_key": key,
        "judge_sample_key": key,
        "job_id": job_id,
        "pair_id": pair_id,
        "model_role": role,
        "attack_source_family": source,
        "eval_model_family": target,
        "benchmark": benchmark,
        "sample_id": f"{benchmark}-{key}",
        "original_id": key,
        "goal": prompt,
        "model_output": output,
        "status": "success",
    }


def _judgement_row(**overrides: object) -> dict[str, object]:
    row = {
        **_generation_row(key="row-1", benchmark="wildjailbreak"),
        "any_harmful_output": True,
        "prompt_matched_attack_success": True,
        "judge_error": None,
    }
    row.update(overrides)
    return row


def _strongreject_args(run_dir: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "run_dir": run_dir,
        "output_dir": None,
        "judge_dir": None,
        "strongreject_rubric": "maintained",
        "judge_model": "fixture-judge",
        "model_roles": None,
        "job_ids": None,
        "pair_ids": None,
        "max_samples": None,
        "resume": False,
        "resume_from": None,
        "restart": False,
        "dry_run": False,
        "strongreject_deferred_retry_rounds": 1,
        "print_every": 0,
        "judge_concurrency": 1,
        "judge_temperature": 0.0,
        "judge_timeout_seconds": 1.0,
        "judge_max_retries": 0,
        "judge_parse_retries": 0,
        "judge_api_retry_sleep_seconds": 0.0,
        "judge_api_retry_forever": False,
        "judge_max_output_tokens": 128,
    }
    values.update(overrides)
    return Namespace(**values)


def _grader_result(score: float = 1.0) -> dict[str, object]:
    return {
        "refusal": 0,
        "convincingness": 5,
        "specificity": 5,
        "score": score,
        "grader_raw_output": "1.b 0\n2.b 5\n3.b 5",
        "grader_error": "",
        "grader_attempts": 1,
        "grader_parse_attempts": 1,
    }


def _harmbench_metadata_rows(
    *,
    standard_count: int = 159,
    include_contextual: bool = True,
) -> list[dict[str, str]]:
    rows = [
        {
            "BehaviorID": f"standard-{index}",
            "Behavior": f"standard prompt {index}",
            "FunctionalCategory": "standard",
        }
        for index in range(standard_count)
    ]
    if include_contextual:
        rows.append(
            {
                "BehaviorID": "contextual-0",
                "Behavior": "context prompt",
                "FunctionalCategory": "contextual",
            }
        )
    return rows


class StrongRejectTests(unittest.TestCase):
    def test_versioned_fingerprints_parser_and_formula(self) -> None:
        self.assertEqual(
            RUBRICS["maintained"].fingerprint,
            EXPECTED_MAINTAINED_FINGERPRINT,
        )
        self.assertEqual(
            RUBRICS["paper_2024"].fingerprint,
            EXPECTED_PAPER_FINGERPRINT,
        )
        parsed = parse_strongreject_judgement(
            "#scores\n1.a reason\n1.b 0\n2.a reason\n2.b 4\n" "3.a reason\n3.b 2\n"
        )
        self.assertEqual(parsed.refusal, 0)
        self.assertEqual(parsed.convincingness, 4)
        self.assertEqual(parsed.specificity, 2)
        self.assertEqual(parsed.score, 0.5)
        self.assertEqual(score_strongreject(1, 5, 5), 0.0)
        with self.assertRaises(ValueError):
            parse_strongreject_judgement("1.b 0\n2.b 6\n3.b 1")

    def test_cache_resume_avoids_duplicate_grader_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generation_dir = root / "generations" / "source"
            write_jsonl(
                generation_dir / "strongreject.jsonl",
                [
                    _generation_row(key="one"),
                    _generation_row(key="two", job_id="source-job-two"),
                ],
            )
            output_dir = root / "reports" / "strongreject"
            with (
                mock.patch(
                    "scripts.recompute_saved_benchmark_scores._ReloadableClient"
                ),
                mock.patch(
                    "scripts.recompute_saved_benchmark_scores._invoke_strongreject",
                    return_value=_grader_result(),
                ) as invoke,
            ):
                first = run_strongreject(_strongreject_args(root), output_dir)
            # Identical prompt/response pairs share one content-addressed call.
            self.assertEqual(invoke.call_count, 1)
            self.assertEqual(first["overall"]["scored_count"], 2)

            with mock.patch(
                "scripts.recompute_saved_benchmark_scores._invoke_strongreject",
                side_effect=AssertionError("resume must use the cache"),
            ) as resumed_invoke:
                second = run_strongreject(
                    _strongreject_args(root, resume=True), output_dir
                )
            resumed_invoke.assert_not_called()
            self.assertEqual(second["overall"]["scored_count"], 2)

    def test_failed_call_is_recovered_by_deferred_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(
                root / "generations" / "source" / "strongreject.jsonl",
                [_generation_row(key="one")],
            )
            with (
                mock.patch(
                    "scripts.recompute_saved_benchmark_scores._ReloadableClient"
                ),
                mock.patch(
                    "scripts.recompute_saved_benchmark_scores._invoke_strongreject",
                    side_effect=[RuntimeError("temporary failure"), _grader_result()],
                ) as invoke,
            ):
                summary = run_strongreject(_strongreject_args(root), root / "reports")
            self.assertEqual(invoke.call_count, 2)
            self.assertEqual(summary["initial_deferred_error_count"], 1)
            self.assertEqual(summary["recovered_after_deferred_retry_count"], 1)
            self.assertEqual(summary["unresolved_after_deferred_retry_count"], 0)


class DiscoveryAndSummaryTests(unittest.TestCase):
    def test_generation_and_judgement_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = _generation_row(key="same")
            write_jsonl(
                root / "generations" / "a" / "strongreject.jsonl",
                [row, row],
            )
            generations = load_generation_rows(root, benchmark="strongreject")
            self.assertEqual(len(generations), 1)

            failed = {
                **row,
                "any_harmful_output": None,
                "prompt_matched_attack_success": None,
                "judge_error": "temporary",
            }
            successful = {
                **row,
                "any_harmful_output": True,
                "prompt_matched_attack_success": True,
                "judge_error": None,
            }
            write_jsonl(root / "judge" / "a.judgements.jsonl", [failed])
            write_jsonl(root / "judge" / "b.judgements.jsonl", [successful])
            judgements, paths = load_judgements(root / "judge")
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(judgements), 1)
            self.assertTrue(judgements[0]["prompt_matched_attack_success"])

    def test_shared_summary_matches_existing_rigj_summary(self) -> None:
        rows = [
            {
                **_generation_row(key="source"),
                "any_harmful_output": True,
                "prompt_matched_attack_success": True,
            },
            {
                **_generation_row(
                    key="target",
                    job_id="target-job",
                    pair_id="qwen_to_llama",
                    role="target",
                    target="llama",
                ),
                "any_harmful_output": False,
                "prompt_matched_attack_success": False,
            },
        ]
        manifests = {
            "source-job": {
                "job_id": "source-job",
                "attack_source_family": "qwen",
                "eval_family": "qwen",
                "summaries": {"strongreject": {"selected_count": 2}},
            },
            "target-job": {
                "job_id": "target-job",
                "attack_source_family": "qwen",
                "eval_family": "llama",
                "summaries": {"strongreject": {"selected_count": 2}},
            },
        }
        expected = _summarize(rows, manifests)
        actual = summarize_judgements(rows, manifests)
        expected.pop("created_at")
        actual.pop("created_at")
        self.assertEqual(actual, expected)


class HarmBenchTests(unittest.TestCase):
    def test_official_metadata_filters_and_uses_judged_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            judge_path = root / "judge.judgements.jsonl"
            write_jsonl(
                judge_path,
                [
                    {
                        **_judgement_row(
                            generation_key="hb-1",
                            judge_sample_key="hb-1",
                            benchmark="harmbench",
                            sample_id="harmbench-00001",
                            original_id="standard-0",
                            goal="standard prompt 0",
                        )
                    },
                    {
                        **_judgement_row(
                            generation_key="hb-2",
                            judge_sample_key="hb-2",
                            benchmark="harmbench",
                            sample_id="harmbench-00002",
                            original_id="standard-1",
                            goal="standard prompt 1",
                            any_harmful_output=None,
                            prompt_matched_attack_success=None,
                            judge_error="fixture error",
                        )
                    },
                    {
                        **_judgement_row(
                            generation_key="hb-3",
                            judge_sample_key="hb-3",
                            benchmark="harmbench",
                            sample_id="harmbench-00003",
                            original_id="not-the-official-id",
                            goal="standard prompt 2",
                            any_harmful_output=False,
                            prompt_matched_attack_success=False,
                        )
                    },
                    {
                        **_judgement_row(
                            generation_key="hb-4",
                            judge_sample_key="hb-4",
                            benchmark="harmbench",
                            sample_id="harmbench-00004",
                            original_id="contextual-0",
                            goal="context prompt",
                        )
                    },
                    {
                        **_judgement_row(
                            generation_key="hb-5",
                            judge_sample_key="hb-5",
                            benchmark="harmbench",
                            sample_id="harmbench-00005",
                            original_id="not-in-official-test",
                            goal="training-only prompt",
                        )
                    },
                ],
            )
            metadata = root / "harmbench.jsonl"
            write_jsonl(metadata, _harmbench_metadata_rows())
            args = Namespace(
                run_dir=root,
                judge_dir=judge_path,
                harmbench_metadata=metadata,
                dry_run=False,
            )
            summary = run_harmbench_standard(args, root / "report")
            self.assertEqual(summary["category_counts"]["standard"], 3)
            self.assertEqual(summary["excluded_contextual_count"], 1)
            self.assertEqual(summary["excluded_not_in_official_test_count"], 1)
            self.assertEqual(summary["excluded_nonstandard_count"], 2)
            self.assertEqual(summary["overall"]["selected_count"], 3)
            self.assertEqual(summary["overall"]["judged_count"], 2)
            self.assertEqual(summary["overall"]["judge_error_count"], 1)
            self.assertEqual(summary["overall"]["asr"], 0.5)
            self.assertEqual(summary["match_method_counts"]["original_id"], 3)
            self.assertEqual(summary["match_method_counts"]["prompt_exact"], 1)
            self.assertEqual(summary["match_method_counts"]["not_in_official_test"], 1)
            self.assertEqual(
                summary["harmbench_metadata"]["category_counts"]["standard"], 159
            )
            self.assertEqual(
                len(read_jsonl(root / "report" / "harmbench_standard_rows.jsonl")),
                3,
            )

    def test_default_metadata_path_is_strict_official_whitelist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = root / "harmbench_behaviors_text_test.jsonl"
            write_jsonl(metadata, _harmbench_metadata_rows())
            with mock.patch(
                "scripts.recompute_saved_benchmark_scores."
                "DEFAULT_HARMBENCH_METADATA_PATH",
                metadata,
            ):
                by_id, by_prompt, info = _harmbench_indexes(
                    Namespace(run_dir=root, harmbench_metadata=None)
                )
            self.assertEqual(by_id["standard-0"][0]["category"], "standard")
            self.assertIn(normalize_prompt("standard prompt 0"), by_prompt)
            self.assertEqual(info["path"], str(metadata.resolve()))
            self.assertEqual(info["category_counts"]["standard"], 159)

    def test_metadata_without_159_standard_rows_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = root / "invalid.jsonl"
            write_jsonl(metadata, _harmbench_metadata_rows(standard_count=158))
            with self.assertRaisesRegex(ValueError, "exactly 159 standard rows"):
                _harmbench_indexes(Namespace(run_dir=root, harmbench_metadata=metadata))

    def test_official_id_prompt_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            judge_path = root / "judge.judgements.jsonl"
            write_jsonl(
                judge_path,
                [
                    _judgement_row(
                        generation_key="hb-mismatch",
                        judge_sample_key="hb-mismatch",
                        benchmark="harmbench",
                        original_id="standard-0",
                        goal="wrong prompt",
                    )
                ],
            )
            metadata = root / "harmbench.jsonl"
            write_jsonl(metadata, _harmbench_metadata_rows())
            args = Namespace(
                run_dir=root,
                judge_dir=judge_path,
                harmbench_metadata=metadata,
                dry_run=False,
            )
            with self.assertRaisesRegex(ValueError, "prompt mismatch"):
                run_harmbench_standard(args, root / "report")


class WildJailbreakAuditTests(unittest.TestCase):
    @staticmethod
    def _record(prompt: str, label: str, index: int) -> SourceRecord:
        return SourceRecord(
            dataset="wildjailbreak",
            row_index=index,
            sample_id=f"source-{index}",
            prompt=prompt,
            label=label,
            source_path="fixture.arrow",
        )

    def test_benign_harmful_and_unknown_are_distinguished(self) -> None:
        benign = self._record("benign prompt", "adversarial_benign", 1)
        harmful = self._record("harmful prompt", "adversarial_harmful", 2)
        by_prompt = {
            normalize_prompt(benign.prompt): [benign],
            normalize_prompt(harmful.prompt): [harmful],
        }
        rows = [
            annotate_row(
                _judgement_row(goal="benign prompt", judge_sample_key="benign"),
                by_id={},
                by_prompt=by_prompt,
            ),
            annotate_row(
                _judgement_row(goal="harmful prompt", judge_sample_key="harmful"),
                by_id={},
                by_prompt=by_prompt,
            ),
            annotate_row(
                _judgement_row(goal="missing prompt", judge_sample_key="unknown"),
                by_id={},
                by_prompt=by_prompt,
            ),
        ]
        self.assertEqual(
            [row["benign_audit_status"] for row in rows],
            ["benign", "harmful", "unknown"],
        )
        corrected, metadata = corrected_asr_rows(rows)
        self.assertEqual(len(corrected), 2)
        self.assertEqual(metadata["excluded_wildjailbreak_benign_rows"], 1)
        self.assertEqual(metadata["included_wildjailbreak_unknown_rows"], 1)

    def test_audit_writes_all_formats_and_preserves_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            judge_path = root / "fixture.judgements.jsonl"
            write_jsonl(
                judge_path,
                [
                    _judgement_row(
                        generation_key="benign",
                        judge_sample_key="benign",
                        goal="benign prompt",
                    ),
                    _judgement_row(
                        generation_key="unknown",
                        judge_sample_key="unknown",
                        goal="unknown prompt",
                    ),
                    _judgement_row(
                        generation_key="other-benchmark",
                        judge_sample_key="other-benchmark",
                        benchmark="advbench",
                        goal="not audited",
                    ),
                ],
            )
            benign = self._record("benign prompt", "adversarial_benign", 1)
            indexes = ({}, {normalize_prompt(benign.prompt): [benign]}, {})
            args = Namespace(
                judge_output_path=judge_path,
                audit_output_dir=root / "audit",
                audit_run_name="fixture",
                wildjailbreak_arrow=[],
                include_inputs_jsonl=False,
                fail_on_unmatched_wildjailbreak=False,
            )
            with mock.patch(
                "scripts.audit_judge_outputs_benign_samples.build_source_indexes",
                return_value=indexes,
            ):
                summary = run_audit(args)
            self.assertEqual(summary["wildjailbreak_benign_rows"], 1)
            self.assertEqual(summary["wildjailbreak_unknown_rows"], 1)
            corrected = read_jsonl(root / "audit" / "fixture.asr_corrected.rows.jsonl")
            self.assertEqual(
                [row["judge_sample_key"] for row in corrected],
                ["other-benchmark", "unknown"],
            )
            wildjailbreak_only = read_jsonl(
                root / "audit" / "fixture.wildjailbreak_only.rows.jsonl"
            )
            self.assertEqual(
                [row["judge_sample_key"] for row in wildjailbreak_only],
                ["unknown"],
            )
            wildjailbreak_summary = json.loads(
                (root / "audit" / "fixture.wildjailbreak_only.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(wildjailbreak_summary["overall"]["selected_count"], 1)
            self.assertEqual(wildjailbreak_summary["overall"]["judged_count"], 1)
            all_benchmark_summary = json.loads(
                (root / "audit" / "fixture.asr_corrected.summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(all_benchmark_summary["overall"]["selected_count"], 2)
            self.assertEqual(
                all_benchmark_summary["wildjailbreak_only"]["overall"],
                wildjailbreak_summary["overall"],
            )
            self.assertEqual(
                summary["corrected_wildjailbreak_overall"]["selected_count"], 1
            )
            for suffix in (
                "annotated.jsonl",
                "annotated.csv",
                "summary.json",
                "asr_corrected.rows.jsonl",
                "asr_corrected.rows.csv",
                "asr_corrected.summary.json",
                "asr_corrected.summary.md",
                "wildjailbreak_only.rows.jsonl",
                "wildjailbreak_only.rows.csv",
                "wildjailbreak_only.summary.json",
                "wildjailbreak_only.summary.md",
            ):
                self.assertTrue((root / "audit" / f"fixture.{suffix}").is_file())

    def test_ahs_paper_five_filter_excludes_nonstandard_non_ah_and_other_benches(
        self,
    ) -> None:
        def ahs_row(key: str, benchmark: str, **extra: object) -> dict[str, object]:
            return {
                **_judgement_row(
                    generation_key=key,
                    judge_sample_key=key,
                    benchmark=benchmark,
                ),
                "judge_metric": "ahs",
                "harmfulness_score": 5,
                **extra,
            }

        rows = [
            ahs_row("adv", "advbench"),
            ahs_row("hb-standard", "harmbench", harmbench_category="standard"),
            ahs_row(
                "hb-other",
                "harmbench",
                harmbench_category="not_in_official_test",
            ),
            ahs_row("jbb", "jailbreakbench"),
            ahs_row("mi", "malicious_instruct"),
            ahs_row(
                "wj-ah",
                "wildjailbreak",
                benign_audit_label="adversarial_harmful",
            ),
            ahs_row(
                "wj-benign",
                "wildjailbreak",
                benign_audit_label="adversarial_benign",
            ),
            ahs_row("strongreject", "strongreject"),
        ]

        filtered, audit = filter_ahs_paper_five_rows(rows)

        self.assertEqual(
            {row["judge_sample_key"] for row in filtered},
            {"adv", "hb-standard", "jbb", "mi", "wj-ah"},
        )
        self.assertEqual(audit["input_row_count"], 8)
        self.assertEqual(audit["retained_row_count"], 5)
        self.assertEqual(audit["excluded_row_count"], 3)
        summary = _summarize_ahs(filtered)
        summary["filter_audit"] = audit
        markdown = ahs_paper_five_markdown(summary)
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
        self.assertIn("| harmbench-standard | 1 | 1/1 | 0 | 5.000 |", markdown)
        self.assertIn("| benchmark_strongreject | 1 |", markdown)


class CommandLineTests(unittest.TestCase):
    def test_strongreject_cli_dry_run_never_needs_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_jsonl(
                root / "generations" / "source" / "strongreject.jsonl",
                [_generation_row(key="one")],
            )
            output = root / "dry-run"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/recompute_saved_benchmark_scores.py",
                    "--run-dir",
                    str(root),
                    "--mode",
                    "strongreject",
                    "--output-dir",
                    str(output),
                    "--judge-model",
                    "fixture-judge",
                    "--dry-run",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )
            summary = json.loads(
                (
                    output / "strongreject" / "strongreject_dry_run_summary.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(summary["pending_calls"], 1)


if __name__ == "__main__":
    unittest.main()
