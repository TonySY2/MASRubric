"""No-model checks for handoff planning, result identities and failure receipts."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/rebuttal"
sys.path.insert(0, str(SCRIPTS))
import run_matrix
import summarize
import audit_traces
import check_split_overlap

_paper_spec = importlib.util.spec_from_file_location("rebuttal_paper_launcher", SCRIPTS.parents[1] / "test/run_paper_main.py")
paper_launcher = importlib.util.module_from_spec(_paper_spec)
_paper_spec.loader.exec_module(paper_launcher)


class RebuttalMatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            path.write_text("".join(json.dumps(row) + "\n" for row in value), encoding="utf-8")
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def job(self, **extra):
        return dict(id="math_8b__gsm8k__masrubric__r01", benchmark="gsm8k", method="masrubric",
                    suite="math_8b", runtime="paper", family="main", expected=2,
                    full_benchmark_total=1319, repeat=1, output_dir=str(self.root / "job"), **extra)

    def paper_results(self, rows, ids=None):
        result = self.write("job/task/result.json", rows)
        if ids is not None:
            self.write("job/task/manifest.json", {"sample_ids": ids})
        return result

    def test_denominator_preserves_unknown_and_missing(self):
        job = self.job()
        job["expected"] = 4
        self.paper_results([{"id": 0, "is_correct": True}, {"id": 1, "is_correct": False},
                            {"id": 2, "error": "api_error", "is_correct": False}], [0, 1, 2, 3])
        result = summarize.summarize_job(job)
        self.assertEqual((result["correct"], result["incorrect"], result["unknown"], result["missing"]), (1, 1, 1, 1))
        self.assertEqual(result["accuracy_full_denominator"], .25)
        self.assertFalse(result["complete"])

    def test_wrong_id_cannot_replace_missing_sample(self):
        self.paper_results([{"id": 0, "is_correct": True}, {"id": 99, "is_correct": True}], [0, 1])
        result = summarize.summarize_job(self.job())
        self.assertEqual(result["unexpected_ids"], ["99"])
        self.assertEqual(result["missing_ids"], ["1"])
        self.assertEqual(result["accuracy_full_denominator"], .5)
        self.assertFalse(result["complete"])

    def test_duplicate_result_ids_fail(self):
        self.paper_results([{"id": 0, "is_correct": True}, {"id": 0, "is_correct": False}])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize.summarize_job(self.job())

    def test_functional_failure_and_transport_timeout_differ(self):
        self.assertIs(summarize.outcome({"is_solved": False, "error": "AssertionError: wrong answer"}), False)
        self.assertIsNone(summarize.outcome({"is_solved": False, "error": "sample_timeout"}))
        self.assertIsNone(summarize.outcome({"is_correct": True, "status": "timeout"}))

    def test_legacy_aqua_answers(self):
        self.assertIs(summarize.outcome({"answer": ["A"], "hypothesis": " a "}, "aqua"), True)
        self.assertIs(summarize.outcome({"answer": "A", "hypothesis": "B"}, "aqua"), False)
        self.assertIsNone(summarize.outcome({"answer": "AB", "hypothesis": "AB"}, "aqua"))

    def test_regrading_overrides_any_stored_success_field(self):
        path = self.paper_results([{"id": 0, "is_correct": True, "is_solved": True},
                                   {"id": 1, "is_solved": False}], [0, 1])
        self.write("job/task/paper_grader.json", {"trust_stored_flags": False, "total_records": 2,
                   "matched_records": 2, "correct": 0, "missing_dataset_records": [],
                   "mismatches": [{"id": 0, "stored": True, "rejudged": False}]})
        self.assertIs(summarize.outcome(summarize.indexed(path)["0"]), False)
        job = self.job()
        job["benchmark"] = "mbpp"
        result = summarize.summarize_job(job)
        self.assertEqual(result["correct"], 0)
        self.assertTrue(result["complete"])

    def test_regrading_unmatched_sample_stays_unknown(self):
        self.paper_results([{"id": 0, "is_solved": True}, {"id": 1, "is_solved": True}], [0, 1])
        self.write("job/task/paper_grader.json", {"trust_stored_flags": False, "total_records": 2,
                   "matched_records": 1, "correct": 1, "missing_dataset_records": [1], "mismatches": []})
        job = self.job()
        job["benchmark"] = "mbpp"
        result = summarize.summarize_job(job)
        self.assertEqual((result["correct"], result["unknown"]), (1, 1))
        self.assertFalse(result["complete"])

    def test_missing_reference_code_grader_is_incomplete(self):
        self.paper_results([{"id": 0, "is_solved": True}, {"id": 1, "is_solved": False}], [0, 1])
        job = self.job()
        job["benchmark"] = "mbpp"
        result = summarize.summarize_job(job)
        self.assertFalse(result["complete"])
        self.assertEqual(result["grader_validation"], "missing")

    def test_invalid_grader_count_is_rejected(self):
        path = self.paper_results([{"id": 0, "is_solved": True}])
        self.write("job/task/paper_grader.json", {"trust_stored_flags": False, "total_records": 2,
                   "matched_records": 2, "correct": 2, "mismatches": []})
        with self.assertRaisesRegex(ValueError, "count"):
            summarize.indexed(path)

    def test_existing_codecontest_id_suffix_is_normalized(self):
        self.paper_results([{"id": "0_abcdef12", "is_solved": True}, {"id": "1_1234abcd", "is_solved": False}], [0, 1])
        job = self.job()
        job.update(benchmark="codecontest", runtime="release")
        result = summarize.summarize_job(job)
        self.assertTrue(result["complete"])
        self.assertEqual(result["identity_validation"], "passed:runtime_manifest")

    def test_release_humaneval_uses_input_names(self):
        path = self.write("input.jsonl", [{"name": "HumanEval_23_strlen"}, {"name": "HumanEval_24_largest_divisor"}])
        self.paper_results([{"id": "HumanEval_23_strlen", "is_solved": True}, {"id": "HumanEval_24_largest_divisor", "is_solved": False}])
        job = self.job()
        job.update(benchmark="humaneval", runtime="release", dataset_file=str(path))
        self.assertEqual(summarize.summarize_job(job)["identity_validation"], "passed:selected_input_ids")

    def test_input_jsonl_keeps_unicode_line_separator_inside_string(self):
        path = self.root / "input.jsonl"
        path.write_text('{"id": 0, "text": "left\u2028right"}\n{"id": 1}\n', encoding="utf-8")
        job = self.job(dataset_file=str(path))
        ids, source = summarize.expected_ids(job, [])
        self.assertEqual(ids, {"0", "1"})
        self.assertEqual(source, "selected_input_ids")

    def test_nested_amc_input_uses_selected_records(self):
        path = self.write("input.json", {"test": [{"problem": "one"}, {"problem": "two"}]})
        job = self.job(dataset_file=str(path))
        job["benchmark"] = "amc23"
        self.assertEqual(summarize.expected_ids(job, [])[0], {"0", "1"})

    def test_usage_completeness_is_read_from_buckets(self):
        self.paper_results([{"id": 0, "is_correct": True}, {"id": 1, "is_correct": False}], [0, 1])
        self.write("job/task/run.usage.summary.json", {"token_usage": {
                   "llm": {"complete": True, "observed_total_tokens": 17},
                   "embedding": {"complete": True, "observed_total_tokens": 3}}})
        result = summarize.summarize_job(self.job())
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["llm_total_tokens"], 17)

    def test_groups_do_not_mix_model_runtime_or_denominator(self):
        base = dict(self.job(), complete=True, status="completed", accuracy_full_denominator=.5)
        rows = [base, dict(base, suite="math_14b"), dict(base, expected=1),
                dict(base, runtime="release"), dict(base, family="sensitivity"), dict(base, full_benchmark_total=1400)]
        groups = summarize.summarize_groups(rows)
        self.assertEqual(len(groups), 6)
        self.assertTrue(all(group["runs_complete"] == 1 for group in groups))

    def test_large_exact_mcnemar_does_not_overflow(self):
        left = self.write("left.jsonl", [{"id": i, "correct": i % 2 == 0} for i in range(1319)])
        right = self.write("right.jsonl", [{"id": i, "correct": i % 2 != 0} for i in range(1319)])
        result = summarize.paired(left, right, iterations=100)
        self.assertTrue(math.isfinite(result["mcnemar_exact_two_sided_p"]))
        self.assertAlmostEqual(result["mcnemar_exact_two_sided_p"], 1.0)
        self.assertEqual(result["paired_scored"], 1319)

    def test_paired_missing_and_unknown_are_explicit(self):
        left = self.write("left.jsonl", [{"id": 0, "correct": False}, {"id": 1, "status": "timeout"}, {"id": 2, "correct": True}])
        right = self.write("right.jsonl", [{"id": 0, "correct": True}, {"id": 1, "correct": True}, {"id": 3, "correct": True}])
        result = summarize.paired(left, right, iterations=100)
        self.assertEqual((result["paired_scored"], result["paired_unknown"], result["left_only"], result["right_only"]), (1, 1, 1, 1))

    def test_resume_requires_matching_job_and_verified_results(self):
        job = self.job(command=["python", "runner"])
        previous = dict(job, status="completed")
        self.assertTrue(run_matrix.validate_resume(job, previous, lambda _: {"complete": True}))
        with self.assertRaisesRegex(ValueError, "configuration"):
            run_matrix.validate_resume(job, dict(previous, expected=3), lambda _: {"complete": True})
        with self.assertRaisesRegex(ValueError, "no longer"):
            run_matrix.validate_resume(job, previous, lambda _: {"complete": False})

    def execute_fixture(self):
        job = self.job(command=[sys.executable, str(run_matrix.ROOT / "test/run_paper_main.py")])
        plan = {"runtime": "paper", "family": "main", "endpoint_variables": [], "jobs": [job]}
        args = argparse.Namespace(action="run", resume=False, timeout=1, keep_going=False)
        return plan, args

    def test_launch_error_writes_failed_receipt(self):
        plan, args = self.execute_fixture()
        with patch.object(run_matrix.subprocess, "Popen", side_effect=OSError("missing interpreter")):
            self.assertEqual(run_matrix.execute(plan, args), 1)
        receipt = json.loads((self.root / "job/job.json").read_text())
        self.assertEqual((receipt["status"], receipt["failure_kind"]), ("failed", "OSError"))

    def test_summary_error_cannot_leave_completed_receipt(self):
        plan, args = self.execute_fixture()
        proc = MagicMock()
        proc.wait.return_value = 0
        proc.poll.return_value = 0
        with patch.object(run_matrix.subprocess, "Popen", return_value=proc), patch.object(summarize, "summarize_job", side_effect=ValueError("duplicate id")):
            self.assertEqual(run_matrix.execute(plan, args), 1)
        receipt = json.loads((self.root / "job/job.json").read_text())
        self.assertEqual((receipt["status"], receipt["failure_kind"]), ("failed", "ValueError"))

    def test_timeout_stops_owned_child_and_records_timeout(self):
        plan, args = self.execute_fixture()
        proc = MagicMock()
        proc.wait.side_effect = subprocess.TimeoutExpired("fixture", 1)
        with patch.object(run_matrix.subprocess, "Popen", return_value=proc), patch.object(run_matrix, "stop_child") as stop:
            self.assertEqual(run_matrix.execute(plan, args), 1)
            stop.assert_called_once_with(proc)
        self.assertEqual(json.loads((self.root / "job/job.json").read_text())["status"], "timeout")

    def test_existing_directory_is_never_overwritten(self):
        plan, args = self.execute_fixture()
        marker = self.write("job/retained.json", {"value": 123})
        with self.assertRaises(FileExistsError):
            run_matrix.execute(plan, args)
        self.assertEqual(json.loads(marker.read_text()), {"value": 123})

    def test_plan_preserves_suite_family_and_full_denominator(self):
        config = self.write("matrix.json", {"families": {"main": {"domains": ["math"], "runtime": "paper", "methods": ["baseline", "masrubric"], "interpretation": "test"}}})
        paper = self.write("paper.json", {"benchmarks": {"gsm8k": {"domain": "math", "total": 1319, "dataset": "datasets/gsm8k/gsm8k_id.jsonl"}}})
        args = argparse.Namespace(family="main", suite="math_14b", benchmarks="all", repeats=2,
                                  output_dir=self.root / "outputs", concurrency=1, assets_root=None,
                                  olympiad_python=None, allow_model_override=False, limit=2)
        with patch.object(run_matrix, "CONFIG", config), patch.object(run_matrix, "PAPER", paper):
            plan = run_matrix.make_plan(args)
        self.assertEqual(len(plan["jobs"]), 4)
        for job in plan["jobs"]:
            self.assertEqual((job["suite"], job["family"], job["expected"], job["full_benchmark_total"]), ("math_14b", "main", 2, 1319))
            self.assertEqual(job["repeat_kind"], "fresh_stochastic_run_no_seed_contract")

    def macro_rows(self, suite="math_8b"):
        config = json.loads(run_matrix.PAPER.read_text(encoding="utf-8"))
        domain = "code" if suite == "code_8b" else "math"
        return [dict(suite=suite, runtime="paper", family="main", method="masrubric", repeat=1,
                     benchmark=name, expected=spec["total"], full_benchmark_total=spec["total"],
                     complete=True, status="completed", accuracy_full_denominator=.5)
                for name, spec in config["benchmarks"].items() if spec["domain"] == domain]

    def test_macro_requires_complete_fixed_suite(self):
        rows = self.macro_rows()
        complete = summarize.summarize_macros(rows)[0]
        self.assertTrue(complete["complete"])
        self.assertEqual(len(complete["expected_benchmarks"]), 9)
        self.assertEqual(complete["macro_accuracy_percent"], 50)
        partial = summarize.summarize_macros(rows[:-1])[0]
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["missing_benchmarks"], [rows[-1]["benchmark"]])
        self.assertIsNone(partial["macro_accuracy_percent"])

    def test_macro_rejects_smoke_wrong_denominator_and_duplicates(self):
        rows = self.macro_rows("code_8b")
        rows[0]["expected"] = 2
        smoke = summarize.summarize_macros(rows)[0]
        self.assertIsNone(smoke["macro_accuracy"])
        self.assertEqual(smoke["subset_benchmarks"], [rows[0]["benchmark"]])
        rows = self.macro_rows("code_8b")
        rows[0]["full_benchmark_total"] = 999
        wrong = summarize.summarize_macros(rows)[0]
        self.assertEqual(wrong["invalid_full_denominator_benchmarks"], [rows[0]["benchmark"]])
        rows = self.macro_rows("code_8b")
        duplicate = summarize.summarize_macros(rows + [rows[0]])[0]
        self.assertEqual(duplicate["duplicate_benchmarks"], [rows[0]["benchmark"]])
        self.assertFalse(duplicate["complete"])

    def test_macro_keeps_repeat_model_and_runtime_separate(self):
        rows = self.macro_rows()
        rows += [dict(r, repeat=2) for r in self.macro_rows()]
        rows += self.macro_rows("math_14b")
        rows += [dict(r, runtime="release") for r in self.macro_rows()]
        groups = summarize.summarize_macros(rows)
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(g["complete"] for g in groups))
        rows[-1]["status"] = "failed"
        self.assertEqual(sum(g["complete"] for g in summarize.summarize_macros(rows)), 3)

    def test_audit_missing_and_malformed_judgments_are_not_empty(self):
        path = self.write("audits.json", [{"id": 0, "scores": {
            "empty": {"message": {"content": "x"}, "judgements": []},
            "missing": {"message": {"content": "y"}},
            "malformed": {"message": {"content": "z"}, "judgements": "wrong"}}}])
        counts = Counter()
        records = list(audit_traces.audit_records(path, counts))
        self.assertEqual([r["judgements_state"] for r in records], ["available", "missing", "malformed"])
        self.assertEqual((counts["missing_judgment_messages"], counts["malformed_judgment_messages"]), (1, 1))

    def test_audit_export_blinds_model_verdict_and_preserves_key(self):
        path = self.write("audits.json", [{"id": 0, "question": "a problem", "scores": {
            "message1": {"message": {"content": "candidate answer"}, "role": "solver",
                         "judgements": [{"metric": "angle", "verdict": "Flawed"}], "is_pruned": True}}}])
        output = self.root / "annotations"
        with patch.object(sys, "argv", ["audit_traces", str(path), "--output-dir", str(output)]):
            audit_traces.main()
        blind = json.loads((output / "annotation_blind.json").read_text())
        key = json.loads((output / "annotation_key.private.json").read_text())
        self.assertNotIn("judgements", blind[0])
        self.assertNotIn("withheld", blind[0])
        self.assertEqual(key[0]["judgements"][0]["verdict"], "Flawed")
        self.assertEqual(blind[0]["annotation_id"], key[0]["annotation_id"])

    def test_split_overlap_normalization_and_text_coverage(self):
        path = self.write("split.jsonl", [{"id": 0, "question": "Ａ   test"}, {"id": 1, "prompt": "A test"}, {"id": 2, "answer": 5}])
        questions, summary = check_split_overlap.collect([path])
        self.assertEqual(len(questions), 1)
        self.assertEqual(len(questions["A test"]), 2)
        self.assertEqual(summary["missing_problem_text"], 1)

    def test_main_launcher_rejects_wrong_result_ids(self):
        result = self.write("result.json", [{"id": 0, "is_correct": True}, {"id": 9, "is_correct": True}])
        manifest = dict(result_file=str(result), benchmark="gsm8k", selected_samples=2,
                        sample_ids=["0", "1"], benchmark_total=2, limit=None)
        summary = paper_launcher.result_summary(manifest, 0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["identity_validation"], "failed")
        self.assertEqual(summary["accuracy_percent"], 50)
        self.assertEqual(summary["missing_sample_ids"], ["1"])

    def test_main_launcher_preserves_runtime_unknown(self):
        result = self.write("result.jsonl", [{"id": 0, "is_solved": True}, {"id": 1, "is_solved": False, "error": "sample_timeout"}])
        manifest = dict(result_file=str(result), benchmark="mbpp", selected_samples=2,
                        sample_ids=["0", "1"], benchmark_total=2, limit=None)
        summary = paper_launcher.result_summary(manifest, 0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["unknown_samples"], 1)


if __name__ == "__main__":
    unittest.main()
