import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("single_bank", ROOT / "scripts/rebuttal/single_bank.py")


class SingleBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(cls.module)

    def test_problem_projection_never_contains_gold(self):
        row = dict(id=0, problem="1+1?", question="1+1?", options=["A) 2", "B) 3"],
                   solution="SECRET_SOLUTION", answer="SECRET_ANSWER", correct="SECRET_CHOICE", rationale="SECRET_RATIONALE")
        for benchmark in ("math500", "aqua"):
            self.assertNotIn("SECRET", self.module.task_text(row, benchmark))
        self.assertEqual(self.module.task_text(row, "aqua"), "1+1? Choices: A) 2 B) 3")

    def test_usage_marks_missing_and_failed_calls_unknown(self):
        complete = dict(role="reasoning", status="ok", usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5))
        self.assertEqual(self.module.account_usage([complete])["llm_total_tokens"], 5)
        for bad in (dict(role="supervisor", status="error"), dict(role="reasoning", status="ok", usage=None)):
            self.assertFalse(self.module.account_usage([complete, bad])["complete"])

    def test_model_configuration_failure_leaves_receipt_without_model_requests(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.jsonl"
            data.write_text(json.dumps(dict(id=0, problem="1+1?", answer="2")), encoding="utf-8")
            with patch.dict(os.environ, {"REASONING_MODEL": "wrong-model"}):
                code = self.module.main(["--execute", "--benchmark", "math500", "--method", "full",
                                         "--in-file", str(data), "--output-dir", str(root / "output")])
            self.assertEqual(code, 1)
            receipt = json.loads((root / "output/receipt.json").read_text())
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["missing_ids"], ["0"])
            self.assertEqual(receipt["error_type"], "ValueError")

    def test_actual_reference_solver_and_supervisor_three_conditions(self):
        import test_paper_runtime_integration as fixtures
        from test_release_integration import finding
        import re

        class Endpoint(fixtures.PaperModelEndpoint):
            def __init__(self, benchmark, **kwargs):
                super().__init__(**kwargs)
                self.benchmark = benchmark

            def reply(self, path, body):
                result = super().reply(path, body)
                if self.calls[-1]["stage"] == "batch_audit":
                    names = re.findall(r"^\d+\. Metric Name: (.+)$", body["messages"][-1]["content"], re.MULTILINE)
                    result["choices"][0]["message"]["content"] = json.dumps(
                        {"judgements": [finding(flawed=True, metric=name) for name in names]})
                if self.benchmark == "aqua" and self.calls[-1]["stage"] == "reasoning":
                    result["choices"][0]["message"]["content"] = "The answer is A"
                return result

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, bank, cache = root / "data.jsonl", root / "bank.json", root / "cache.jsonl"
            dataset.write_text(json.dumps(dict(id=0, problem="1+1?", question="1+1?", options=["A) 2", "B) 3"], correct="A", rationale="SECRET_RATIONALE", solution=r"SECRET_SOLUTION \boxed{2}", answer="2")) + "\n", encoding="utf-8")
            metrics = [fixtures.metric("a"), fixtures.metric("b")]
            bank.write_text(json.dumps(metrics), encoding="utf-8")
            cache.write_text("\n".join(json.dumps(dict(name=m["name"], vector=[1., 0.])) for m in metrics), encoding="utf-8")
            counts = {}
            for benchmark in ("math500", "aqua"):
                for method in ("vanilla", "full", "revise-only"):
                    with Endpoint(benchmark) as endpoint:
                        env = dict(os.environ, PYTHONIOENCODING="utf-8")
                        for role in ("REASONING", "SUPERVISOR", "EMBEDDING"):
                            env[role+"_URL"] = endpoint.url
                            env[role+"_MODEL"] = "fixture"
                        out = root / benchmark / method
                        command = [sys.executable, str(ROOT / "scripts/rebuttal/single_bank.py"), "--execute",
                                   "--benchmark", benchmark, "--method", method, "--in-file", str(dataset),
                                   "--metric-pool-file", str(bank), "--embedding-cache-file", str(cache),
                                   "--output-dir", str(out), "--allow-model-override"]
                        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, encoding="utf-8", timeout=70)
                        self.assertEqual(completed.returncode, 0, (completed.stdout + completed.stderr)[-10000:])
                        row = json.loads((out / "result.json").read_text())["0"]
                        self.assertEqual(row["status"], "abstained" if method == "full" else "completed")
                        self.assertEqual(row["is_correct"], method != "full")
                        self.assertTrue(row["usage"]["complete"])
                        self.assertEqual(row["usage"]["llm_total_tokens"], endpoint.total())
                        calls = [json.loads(line) for line in (out / "samples/00000/api_calls.jsonl").read_text().splitlines()]
                        self.assertEqual(len(calls), len(endpoint.calls))
                        self.assertNotIn("SECRET_SOLUTION", json.dumps(calls))
                        counts[benchmark, method] = sum(c["stage"] == "reasoning" for c in endpoint.calls)
                        self.assertFalse(any(c["stage"] == "selector" for c in endpoint.calls))
                        if method != "vanilla":
                            self.assertTrue(any(c["stage"] == "rerank" for c in endpoint.calls))
                            self.assertEqual(len(next(iter(row["scores"].values()))["audit_attempts"]), 4)
                        budget = json.loads((out / "sampling-budget.json").read_text())
                        self.assertEqual(budget["budgets"]["0"], endpoint.total())
            self.assertEqual(counts, {(benchmark, method): calls for benchmark in ("math500", "aqua")
                                      for method, calls in (("vanilla", 1), ("full", 4), ("revise-only", 4))})
            # Missing usage must preserve the actual response and leave an
            # unknown result, with no fabricated zero-token matching budget.
            with Endpoint("math500", missing_usage=True) as endpoint:
                for role in ("REASONING", "SUPERVISOR", "EMBEDDING"):
                    env[role+"_URL"] = endpoint.url
                out = root / "missing-usage"
                command = [sys.executable, str(ROOT / "scripts/rebuttal/single_bank.py"), "--execute",
                           "--benchmark", "math500", "--method", "vanilla", "--in-file", str(dataset),
                           "--metric-pool-file", str(bank), "--embedding-cache-file", str(cache),
                           "--output-dir", str(out), "--allow-model-override"]
                completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, encoding="utf-8", timeout=30)
                self.assertEqual(completed.returncode, 1)
                row = json.loads((out / "result.json").read_text())["0"]
                self.assertEqual(row["status"], "unknown")
                self.assertIsNone(row["is_correct"])
                self.assertFalse(row["usage"]["complete"])
                self.assertEqual(json.loads((out / "sampling-budget.json").read_text())["budgets"], {})
                self.assertIn("response", (out / "samples/00000/api_calls.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
