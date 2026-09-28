"""Offline contracts for the recovered main-table launcher (no SDK or API needed).

The parser checks replay the real runners' argparse declarations without importing
their model dependencies. Separate integration tests exercise the actual SDKs.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_main_launcher_under_test", ROOT / "test/run_paper_main.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def endpoint_environment():
    environment = {}
    for role in launcher.ROLES:
        environment[f"{role.upper()}_URL"] = "http://127.0.0.1:1/v1"
        environment[f"{role.upper()}_MODEL"] = f"test-{role}-model"
        environment[f"{role.upper()}_KEY"] = f"private-test-{role}-key"
    return environment


def runner_parser(script):
    """Use each runner's actual flag declarations while avoiding SDK imports."""
    tree = ast.parse(Path(script).read_text(encoding="utf-8"), filename=str(script))
    declarations = sorted(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
         and isinstance(node.func.value, ast.Name) and node.func.value.id == "parser"
         and node.func.attr == "add_argument"),
        key=lambda node: node.lineno,
    )
    if not declarations:
        raise AssertionError(f"No argparse declarations found in {script}")
    parser = argparse.ArgumentParser()
    namespace = {"parser": parser, "os": os}
    # A few runners use module-level endpoint defaults in their declarations.
    # Evaluate only those constants, never model imports or runner initialization.
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.startswith("DEFAULT_SELECTOR_")
            for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), namespace)
    for declaration in declarations:
        expression = ast.Expression(body=declaration)
        eval(compile(expression, str(script), "eval"), namespace)
    return parser


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.config = launcher.read_json(launcher.CONFIG)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def plan(self, benchmark="gsm8k", suite="math_8b", extra=()):
        args = launcher.parser().parse_args([
            "--suite", suite, "--benchmark", benchmark, "--assets-root", str(self.root / "assets"),
            *extra,
        ])
        return args, launcher.build_plan(args, self.config, benchmark, self.root / "output")

    def test_all_thirteen_tasks_and_both_math_suites_match_real_runner_flags(self):
        self.assertEqual(len(self.config["benchmarks"]), 13)
        self.assertEqual(sum(s["domain"] == "math" for s in self.config["benchmarks"].values()), 9)
        with patch.dict(os.environ, endpoint_environment()):
            for suite in ("math_8b", "math_14b", "code_8b"):
                for benchmark, spec in self.config["benchmarks"].items():
                    if (suite == "code_8b") != (spec["domain"] == "code"):
                        continue
                    for method in ("masrubric", "baseline"):
                        with self.subTest(suite=suite, benchmark=benchmark, method=method):
                            _, (cmd, _, manifest, missing) = self.plan(
                                benchmark, suite, ("--method", method, "--limit", "2", "--concurrency", "3"))
                            parser = runner_parser(launcher.RUNTIME / spec["script"])
                            parsed = parser.parse_args(cmd[cmd.index("--in_file"):])
                            self.assertEqual(parsed.max_turns, 7)
                            self.assertEqual(parsed.concurrency_limit, 3)
                            self.assertEqual(parsed.limit, 2)
                            self.assertEqual(parsed.baseline_only, method == "baseline")
                            self.assertEqual(parsed.pass_rate, 1.0 if spec["domain"] == "code" else 0.6)
                            self.assertEqual(Path(parsed.in_file), Path(manifest["dataset"]))
                            self.assertEqual(missing, [])
                            self.assertEqual(manifest["final_decision"], spec["final_decision"])
                            if spec["domain"] == "code":
                                self.assertEqual(parsed.task, benchmark)
                                self.assertTrue(parsed.force_direct_search)
                                self.assertEqual(parsed.direct_k, 3)
                            else:
                                self.assertEqual((parsed.retrieve_p, parsed.select_q, parsed.retries_times), (20, 5, 3))

    def test_runtime_environment_isolated_and_only_documented_toggles_survive(self):
        environment = dict(endpoint_environment(), PYTHONPATH=str(ROOT / "test"),
                           MASRUBRIC_CHEAP_PRECHECK="1", MASRUBRIC_RANDOM_K_MIN="4",
                           MASRUBRIC_MATH_TEAM_VARIANT="legacy",
                           MASRUBRIC_UNDOCUMENTED_TOGGLE="do-not-inherit")
        with patch.dict(os.environ, environment):
            for benchmark, suite in (("gsm8k", "math_8b"), ("mbpp", "code_8b")):
                with self.subTest(suite=suite):
                    _, (_, env, manifest, _) = self.plan(benchmark, suite)
                    self.assertEqual(Path(env["PYTHONPATH"]), launcher.RUNTIME)
                    self.assertNotIn("MASRUBRIC_UNDOCUMENTED_TOGGLE", env)
                    self.assertEqual(env["MASRUBRIC_CHEAP_PRECHECK"], "0")
                    self.assertEqual(env["MASRUBRIC_RANDOM_K_MIN"], "0")
                    self.assertEqual(env["MASRUBRIC_EXACT_SELECT_Q"], "0")
                    self.assertEqual(env["MASRUBRIC_BATCH_AUDIT_METRICS"], "1")
                    if suite == "math_8b":
                        self.assertEqual(env["MASRUBRIC_MATH_TEAM_VARIANT"], "verifier")
                    else:
                        self.assertEqual(env["MASRUBRIC_PROFILE_AWARE_RETRIEVAL"], "1")
                    self.assertNotIn("do-not-inherit", json.dumps(manifest))

    def test_authentication_is_forwarded_but_never_written_to_manifest(self):
        environment = endpoint_environment()
        with patch.dict(os.environ, environment):
            _, (cmd, env, manifest, _) = self.plan()
        self.assertEqual(cmd[cmd.index("--selector_api_key") + 1], environment["SELECTOR_KEY"])
        for role in ("reasoning", "supervisor", "embedding"):
            self.assertEqual(env[f"MASRUBRIC_{role.upper()}_API_KEY"], environment[f"{role.upper()}_KEY"])
        serialized = json.dumps(manifest)
        for role in launcher.ROLES:
            self.assertNotIn(environment[f"{role.upper()}_KEY"], serialized)
        self.assertFalse(any("KEY" in key for key in manifest["fixed_environment"]))

    def test_bootstrap_imports_runtime_even_with_release_package_on_pythonpath(self):
        runtime = self.root / "isolated-runtime"
        (runtime / "masrubric").mkdir(parents=True)
        (runtime / "masrubric/__init__.py").write_text("ORIGIN = 'recovered'\n", encoding="utf-8")
        decoy = self.root / "release-decoy"
        (decoy / "masrubric").mkdir(parents=True)
        (decoy / "masrubric/__init__.py").write_text("raise RuntimeError('release imported')\n", encoding="utf-8")
        spec = self.config["benchmarks"]["gsm8k"]
        script = runtime / spec["script"]
        script.parent.mkdir(parents=True)
        script.write_text("import masrubric; print(masrubric.ORIGIN)\n", encoding="utf-8")
        with patch.object(launcher, "RUNTIME", runtime), \
                patch.dict(os.environ, dict(endpoint_environment(), PYTHONPATH=str(decoy))):
            _, (cmd, env, _, _) = self.plan()
        result = subprocess.run(cmd, cwd=decoy, env=env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "recovered")

    def test_dry_run_all_tasks_never_starts_process_or_creates_results(self):
        output = self.root / "must-remain-absent"
        with patch.object(launcher.subprocess, "run", side_effect=AssertionError("dry-run started a process")), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = launcher.main(["--suite", "math_8b", "--dry-run", "--output-dir", str(output)])
        self.assertEqual(code, 0)
        self.assertFalse(output.exists())
        self.assertIn('"benchmark": "olymmath_easy"', stdout.getvalue())
        self.assertIn('"benchmark": "olymmath_hard"', stdout.getvalue())

    def test_cross_domain_selection_and_shared_input_override_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.main(["--suite", "math_8b", "--benchmark", "mbpp", "--dry-run"])
            with self.assertRaises(SystemExit):
                launcher.main(["--suite", "math_8b", "--in-file", "one.jsonl", "--dry-run"])

    def test_legacy_interpreter_override_applies_only_to_olympiad(self):
        interpreter = self.root / "legacy-env" / "python"
        with patch.dict(os.environ, endpoint_environment()):
            for benchmark in ("olympiad", "gsm8k"):
                with self.subTest(benchmark=benchmark):
                    _, (cmd, _, manifest, _) = self.plan(
                        benchmark, extra=("--olympiad-python", str(interpreter)))
                    expected = str(interpreter.absolute()) if benchmark == "olympiad" else sys.executable
                    self.assertEqual(cmd[0], expected)
                    self.assertEqual(manifest["python"], expected)
                    self.assertEqual(manifest["source_boundary"], self.config["source_boundary"])

    def test_legacy_interpreter_symlink_is_not_replaced_by_its_base_python(self):
        interpreter = self.root / "legacy-env" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        try:
            interpreter.symlink_to(Path(sys.executable).resolve())
        except OSError as exc:
            if os.name == "nt":
                self.skipTest(f"Windows cannot create the interpreter symlink: {exc}")
            raise
        self.assertNotEqual(interpreter.absolute(), interpreter.resolve())
        with patch.dict(os.environ, endpoint_environment()):
            _, (cmd, _, manifest, _) = self.plan(
                "olympiad", extra=("--olympiad-python", str(interpreter)))
        self.assertEqual(cmd[0], str(interpreter.absolute()))
        self.assertEqual(manifest["python"], str(interpreter.absolute()))
        self.assertEqual(manifest["command"][0], str(interpreter.absolute()))

    def test_legacy_venv_symlink_runs_with_its_own_installed_dependency(self):
        # The regression was not cosmetic: resolving venv/bin/python selected
        # the base interpreter, which could no longer import the legacy grader.
        probe = self.root / "symlink-probe"
        try:
            probe.symlink_to(Path(sys.executable).resolve())
        except OSError as exc:
            if os.name == "nt":
                self.skipTest(f"Windows cannot create virtualenv symlinks: {exc}")
            raise
        probe.unlink()
        environment = self.root / "legacy-venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if os.name == "nt" and not interpreter.is_symlink():
            self.skipTest("This Windows interpreter created a copied executable instead of a symlink")
        self.assertTrue(interpreter.is_symlink())
        package_path = subprocess.run(
            [str(interpreter), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True, capture_output=True, timeout=15, check=True,
        )
        site_packages = Path(package_path.stdout.strip())
        self.assertTrue(site_packages.resolve().is_relative_to(environment.resolve()), site_packages)
        site_packages.mkdir(parents=True, exist_ok=True)
        (site_packages / "paper_test_legacy_dependency.py").write_text(
            "VALUE = 'dependency-from-legacy-venv'\n", encoding="utf-8")

        runtime = self.root / "stub-runtime"
        (runtime / "masrubric").mkdir(parents=True)
        (runtime / "masrubric/__init__.py").write_text("", encoding="utf-8")
        script = runtime / self.config["benchmarks"]["olympiad"]["script"]
        script.parent.mkdir(parents=True)
        script.write_text(
            "import paper_test_legacy_dependency; print(paper_test_legacy_dependency.VALUE)\n",
            encoding="utf-8",
        )
        with patch.object(launcher, "RUNTIME", runtime), patch.dict(os.environ, endpoint_environment()):
            _, (cmd, env, manifest, _) = self.plan(
                "olympiad", extra=("--olympiad-python", str(interpreter)))
        self.assertEqual(cmd[0], str(interpreter.absolute()))
        self.assertEqual(manifest["python"], str(interpreter.absolute()))
        completed = subprocess.run(cmd, cwd=runtime, env=env, text=True, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "dependency-from-legacy-venv")


class PreflightAndResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = launcher.parser().parse_args(["--suite", "code_8b", "--benchmark", "mbpp"])
        self.manifest = {
            "suite": "code_8b", "benchmark": "mbpp", "method": "masrubric",
            "dataset": str(self.root / "data.jsonl"),
            "metric_pool": str(self.root / "pool.json"),
            "embedding_cache": str(self.root / "cache.jsonl"),
            "result_file": str(self.root / "result.jsonl"),
            "benchmark_total": 3, "limit": None,
        }
        historical = launcher.read_json(launcher.CONFIG)["benchmarks"]["mbpp"]["historical"]
        self.manifest["historical_reference"] = historical
        self.manifest["models_and_endpoints"] = {
            role: {"model": f"organization/{model}"} for role, model in historical["model_roles"].items()
        }
        self.pool = [{"name": "a", "match_profile": {"interface_shape": "function_api", "io_contract": "return_value", "match_hint": "arithmetic"}},
                     {"name": "b"}]
        Path(self.manifest["dataset"]).write_text(
            "".join(json.dumps({"task_id": f"task-{i}"}) + "\n" for i in range(3)), encoding="utf-8")
        Path(self.manifest["metric_pool"]).write_text(json.dumps(self.pool), encoding="utf-8")
        Path(self.manifest["embedding_cache"]).write_text(
            "".join(json.dumps({"name": n, "vector": [1.0, 0.0]}) + "\n" for n in ("a", "b")), encoding="utf-8")

    def write_results(self, rows):
        Path(self.manifest["result_file"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_preflight_reads_full_input_but_marks_selected_smoke_samples(self):
        self.args.limit = self.manifest["limit"] = 2
        errors = launcher.preflight(self.manifest, [], self.args)
        self.assertEqual(errors, [])
        self.assertEqual(self.manifest["input_records"], 3)
        self.assertEqual(self.manifest["selected_samples"], 2)
        self.assertEqual(self.manifest["sample_ids"], ["task-0", "task-1"])

    def test_jsonl_unicode_line_separators_stay_inside_records_when_streaming(self):
        rows = [{"task_id": f"task-{i}", "problem": "first\u2028second\u2029third\u0085fourth"}
                for i in range(3)]
        path = Path(self.manifest["dataset"])
        path.write_bytes(("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8"))
        self.assertEqual(launcher.records(path), rows)
        with patch.object(Path, "read_text", side_effect=AssertionError("JSONL streaming must not read the whole file")):
            self.assertEqual(list(launcher.iter_records(path)), rows)
        self.args.limit = self.manifest["limit"] = 2
        self.assertEqual(launcher.preflight(self.manifest, [], self.args), [])
        self.assertEqual(self.manifest["input_records"], 3)
        self.assertEqual(self.manifest["selected_samples"], 2)
        self.assertEqual(self.manifest["sample_ids"], ["task-0", "task-1"])

    def test_preflight_rejects_short_input_even_when_a_smoke_limit_was_requested(self):
        self.args.limit = self.manifest["limit"] = 1
        Path(self.manifest["dataset"]).write_text('{"task_id":"one"}\n', encoding="utf-8")
        errors = launcher.preflight(self.manifest, [], self.args)
        self.assertTrue(any("expected full benchmark 3" in e for e in errors), errors)

    def test_wrong_model_requires_explicit_override_and_remains_recorded(self):
        self.manifest["models_and_endpoints"]["reasoning"]["model"] = "Qwen3-14B"
        errors = launcher.preflight(self.manifest, [], self.args)
        self.assertTrue(any("Historical model mapping differs" in e for e in errors), errors)
        self.args.allow_model_override = True
        self.assertEqual(launcher.preflight(self.manifest, [], self.args), [])
        self.assertTrue(any("reasoning" in e for e in self.manifest["model_name_mismatches"]))

    def test_preflight_reports_profile_coverage_without_inventing_metric_labels(self):
        errors = launcher.preflight(self.manifest, [], self.args)
        self.assertEqual(errors, [])
        coverage = self.manifest["pool_profile_coverage"]
        self.assertEqual(coverage["total"], 2)
        self.assertEqual(coverage["with_match_profile"], 1)
        self.assertEqual(coverage["missing_match_profile"], 1)
        self.assertEqual(coverage["with_interface_and_io"], 1)
        self.assertTrue(self.manifest["warnings"])
        self.assertEqual(launcher.read_json(self.manifest["metric_pool"]), self.pool)

    def test_empty_and_partial_profile_dicts_do_not_count_as_interface_coverage(self):
        pool = [{"name": "a", "match_profile": {}},
                {"name": "b", "match_profile": {"interface_shape": "function_api"}}]
        Path(self.manifest["metric_pool"]).write_text(json.dumps(pool), encoding="utf-8")
        self.assertEqual(launcher.preflight(self.manifest, [], self.args), [])
        coverage = self.manifest["pool_profile_coverage"]
        self.assertEqual(coverage["with_match_profile"], 2)
        self.assertEqual(coverage["missing_match_profile"], 0)
        self.assertEqual(coverage["with_interface_and_io"], 0)
        self.assertTrue(self.manifest["warnings"])
        self.assertEqual(launcher.read_json(self.manifest["metric_pool"]), pool)

    def test_preflight_rejects_missing_or_inconsistent_embedding_vectors(self):
        cache = Path(self.manifest["embedding_cache"])
        cache.write_text('{"name":"a","vector":[1.0,0.0]}\n', encoding="utf-8")
        self.assertTrue(any("misses 1" in e for e in launcher.preflight(self.manifest, [], self.args)))
        cache.write_text('{"name":"a","vector":[1.0,0.0]}\n{"name":"b","vector":[1.0]}\n', encoding="utf-8")
        self.assertTrue(any("dimensions" in e for e in launcher.preflight(self.manifest, [], self.args)))

    def test_missing_results_keep_full_denominator_and_cannot_be_complete(self):
        launcher.preflight(self.manifest, [], self.args)
        self.write_results([{"id": "task-0", "is_solved": True}, {"id": "task-1", "is_solved": False}])
        summary = launcher.result_summary(self.manifest, 0)
        self.assertEqual(summary["expected_samples"], 3)
        self.assertEqual(summary["missing_samples"], 1)
        self.assertAlmostEqual(summary["accuracy_percent"], 100 / 3)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["scope"], "full_benchmark")

    def test_smoke_accuracy_uses_selected_denominator_and_is_labeled(self):
        self.args.limit = self.manifest["limit"] = 2
        launcher.preflight(self.manifest, [], self.args)
        self.write_results([{"id": "task-0", "is_solved": True}, {"id": "task-1", "is_solved": False}])
        summary = launcher.result_summary(self.manifest, 0)
        self.assertEqual(summary["accuracy_percent"], 50.0)
        self.assertEqual(summary["benchmark_total"], 3)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["scope"], "smoke_subset")

    def test_duplicate_ids_or_nonzero_process_exit_cannot_claim_completion(self):
        launcher.preflight(self.manifest, [], self.args)
        self.write_results([{"id": "same", "is_solved": False}] * 3)
        duplicate = launcher.result_summary(self.manifest, 0)
        self.assertTrue(duplicate["duplicate_sample_ids"])
        self.assertFalse(duplicate["complete"])
        self.write_results([{"id": f"task-{i}", "is_solved": True} for i in range(3)])
        self.assertFalse(launcher.result_summary(self.manifest, 124)["complete"])

    def test_aqua_keyed_legacy_results_are_graded_from_answer_and_hypothesis(self):
        self.manifest.update(benchmark="aqua", result_file=str(self.root / "result.json"), selected_samples=3,
                             sample_ids=["10", "11", "12"])
        Path(self.manifest["result_file"]).write_text(json.dumps({
            "10": {"answer": "A", "hypothesis": "A"},
            "11": {"answer": "C", "hypothesis": "B"},
        }), encoding="utf-8")
        summary = launcher.result_summary(self.manifest, 0)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["missing_samples"], 1)
        self.assertAlmostEqual(summary["accuracy_percent"], 100 / 3)
        self.assertFalse(summary["complete"])

    def test_code_score_is_replaced_by_paper_grader_and_keeps_full_denominator(self):
        launcher.preflight(self.manifest, [], self.args)
        self.write_results([{"id": "task-0", "is_solved": True}, {"id": "task-1", "is_solved": True}])
        summary = launcher.result_summary(self.manifest, 0)

        def fake_grader(command, **kwargs):
            self.assertEqual(Path(command[1]).name, "paper_code_eval.py")
            self.assertEqual(command[command.index("--dataset-file") + 1], self.manifest["dataset"])
            kwargs["stdout"].write(json.dumps({"correct": 1, "matched_records": 2}))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(launcher.subprocess, "run", side_effect=fake_grader):
            launcher.regrade_code(self.manifest, summary, endpoint_environment(), 15)
        self.assertEqual(summary["correct"], 1)
        self.assertAlmostEqual(summary["accuracy_percent"], 100 / 3)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["grader_matched_records"], 2)

    def test_code_grader_failure_cannot_claim_completion(self):
        launcher.preflight(self.manifest, [], self.args)
        self.write_results([{"id": f"task-{i}", "is_solved": True} for i in range(3)])
        summary = launcher.result_summary(self.manifest, 0)
        self.assertTrue(summary["complete"])
        with patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)):
            launcher.regrade_code(self.manifest, summary, endpoint_environment(), 15)
        self.assertFalse(summary["complete"])
        self.assertIn("grader_error", summary)


if __name__ == "__main__":
    unittest.main()
