"""Exercise real SDK/agent/runner code against a local fake model endpoint.

The endpoint returns deterministic token usage, so assertions compare exported
accounting with independently observed HTTP calls. No external API is contacted.
These smoke tests verify wiring and accounting, not model/benchmark quality.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))


def finding(flawed=False, metric=None):
    result = {
        "evidence_quote": "1 + 1 = 3" if flawed else "N/A",
        "analysis": "Arithmetic error" if flawed else "N/A",
        "suggestion": "Use 2" if flawed else "N/A",
        "impact_assessment": "YES" if flawed else "NO",
        "is_flawed": flawed,
    }
    if metric is not None:
        result["metric"] = metric
    return result


class FakeModelEndpoint:
    def __init__(self, *, fail_audits=0, malformed_batch=0, missing_usage=False, code=False,
                 http_error=False, http_error_after=None):
        self.calls = []
        self.fail_audits = fail_audits
        self.malformed_batch = malformed_batch
        self.missing_usage = missing_usage
        self.code = code
        self.http_error = http_error
        self.http_error_after = http_error_after
        self.audit_count = 0
        self.batch_count = 0

    def reply(self, path, body):
        index = len(self.calls) + 1
        if path.endswith("/embeddings"):
            usage = {"prompt_tokens": 50 + index, "total_tokens": 50 + index}
            response = {
                "object": "list", "model": body["model"],
                "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                         for i, _ in enumerate(body["input"])],
                "usage": usage,
            }
            stage = "embedding"
        else:
            prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages", []))
            stage = "reasoning"
            if "Select an agent to perform task" in prompt:
                content, stage = "Participant_1", "selector"
            elif "extract key features for metric retrieval" in prompt:
                content = json.dumps({"problem_scenario": ["arithmetic"], "agent_action": ["add"]})
                stage = "summary"
            elif "Select the most relevant indicators" in prompt:
                content = json.dumps({"selected_names": ["arithmetic_a", "arithmetic_b"]})
                stage = "rerank"
            elif "auditing one agent output against multiple indicators" in prompt:
                self.batch_count += 1
                content = "42" if self.batch_count <= self.malformed_batch else json.dumps(
                    [finding(metric="arithmetic_a"), finding(metric="arithmetic_b")])
                stage = "batch_audit"
            elif "Area of Concern" in prompt and "Auditor" in prompt:
                self.audit_count += 1
                content = json.dumps(finding(self.audit_count <= self.fail_audits))
                stage = "audit"
            elif self.code:
                content = "```python\ndef add(a, b):\n    return a + b\n```"
            else:
                content = "1 + 1 = 2. The answer is 2. \\boxed{2}"
            usage = {"prompt_tokens": 100 + index, "completion_tokens": 10 + index,
                     "total_tokens": 110 + index * 2}
            response = {
                "id": f"fake-{index}", "object": "chat.completion", "created": 1,
                "model": body["model"], "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"}],
                "usage": usage,
            }
        if self.missing_usage:
            response.pop("usage", None)
        self.calls.append({"path": path, "body": body, "stage": stage,
                           "usage": None if self.missing_usage else usage})
        return response

    def __enter__(self):
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                size = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(size))
                fail_request = fixture.http_error or (fixture.http_error_after is not None
                                                     and len(fixture.calls) >= fixture.http_error_after)
                if fail_request:
                    fixture.calls.append({"path": self.path, "body": request, "stage": "http_error", "usage": None})
                    response = {"error": {"message": "Deliberate fixture error", "type": "invalid_request_error"}}
                else:
                    response = fixture.reply(self.path, request)
                payload = json.dumps(response).encode("utf-8")
                self.send_response(400 if fail_request else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def total(self, kind="llm"):
        return sum(call["usage"]["total_tokens"] for call in self.calls
                   if call["usage"] is not None and
                   (call["stage"] == "embedding") == (kind == "embedding"))


METRICS = [
    {"name": name, "detailed_definition": "Verify arithmetic.",
     "evaluator_prompt": {"trigger_condition": "Arithmetic", "risk_alert": "Check addition"}}
    for name in ["arithmetic_a", "arithmetic_b"]
]


class RealAgentAccountingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.stdout_capture = contextlib.redirect_stdout(self.output)
        self.stdout_capture.__enter__()

    def tearDown(self):
        self.stdout_capture.__exit__(None, None, None)

    @staticmethod
    def supervisor(endpoint, **kwargs):
        import numpy as np
        from masrubric.agents.supervisor_reasoning_pick_metric import Supervisor
        return Supervisor(
            model="gpt-4o-mini", api_key="local-test", base_url=endpoint.url, domain="math",
            embedding_api_key="local-test", embedding_model="embedding-local",
            embedding_api_base=endpoint.url, preloaded_metrics=METRICS,
            preloaded_embeddings=np.array([[1.0, 0.0], [0.9, 0.0]]), **kwargs)

    async def test_actual_participant_retry_audit_and_final_all_count(self):
        from masrubric.agents.math_solver_gsm8k import MathSolverGsm8k
        from masrubric.agents.final_decision import FinalRefer
        from masrubric.usage import usage_scope
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken

        with FakeModelEndpoint(fail_audits=1) as endpoint:
            supervisor = self.supervisor(endpoint, use_simple_audit=True)
            agent = MathSolverGsm8k("gsm8k", "Participant_1", "gpt-4o", "local-test", endpoint.url,
                                   supervisor=supervisor, reflection_time=1, message_history=[])
            final = FinalRefer("DecisionMaker", "gpt-4o", "local-test", endpoint.url, "gsm8k")
            with usage_scope("retry") as ledger:
                response = await agent.on_messages([TextMessage(source="user", content="What is 1+1?")],
                                                    CancellationToken())
                await final.run_decision([response.chat_message], {agent.name: agent.role}, "What is 1+1?")
            summary = ledger.summary()
            self.assertEqual(endpoint.audit_count, 2)
            self.assertEqual(len(endpoint.calls), 5)
            self.assertEqual(summary["call_count"], 5)
            self.assertEqual(summary["llm"]["total_tokens"], endpoint.total())
            self.assertTrue(summary["llm"]["complete"])
            self.assertEqual(summary["by_stage"]["reasoning"]["call_count"], 2)
            self.assertEqual(summary["by_stage"]["supervisor_audit"]["call_count"], 2)
            self.assertEqual(summary["by_stage"]["final_decision"]["call_count"], 1)

    async def test_retrieval_rerank_batch_parse_retry_and_embedding_all_count(self):
        from masrubric.usage import usage_scope
        from autogen_agentchat.messages import TextMessage

        with FakeModelEndpoint(malformed_batch=1) as endpoint:
            supervisor = self.supervisor(endpoint, retrieval_mode="rerank", select_q=2,
                                         retrieve_p=2, batch_audit_metrics=True)
            with usage_scope("batch") as ledger:
                result = await supervisor.judge("What is 1+1?",
                    TextMessage(source="Participant_1", content="2"), attempt_num=1)
            self.assertTrue(result[0])
            self.assertEqual([c["stage"] for c in endpoint.calls],
                             ["summary", "embedding", "rerank", "batch_audit", "batch_audit"])
            summary = ledger.summary()
            self.assertEqual(summary["call_count"], 5)
            self.assertEqual(summary["llm"]["total_tokens"], endpoint.total())
            self.assertEqual(summary["embedding"]["total_tokens"], endpoint.total("embedding"))
            self.assertEqual(summary["by_stage"]["supervisor_summary"]["call_count"], 1)
            self.assertEqual(summary["by_stage"]["supervisor_rerank"]["call_count"], 1)
            self.assertEqual(summary["by_stage"]["supervisor_audit_batch"]["call_count"], 2)
            self.assertEqual(summary["online_rectifier_llm"]["call_count"], 4)

    async def test_missing_usage_does_not_crash_participant_or_final(self):
        from masrubric.agents.math_solver_gsm8k import MathSolverGsm8k
        from masrubric.agents.final_decision import FinalRefer
        from masrubric.usage import usage_scope
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken

        with FakeModelEndpoint(missing_usage=True) as endpoint:
            agent = MathSolverGsm8k("gsm8k", "Participant_1", "gpt-4o", "local-test", endpoint.url,
                                   reflection_time=0, message_history=[])
            final = FinalRefer("DecisionMaker", "gpt-4o", "local-test", endpoint.url, "gsm8k")
            with usage_scope("unknown") as ledger:
                answer = await agent.on_messages([TextMessage(source="user", content="1+1?")], CancellationToken())
                await final.run_decision([answer.chat_message], {agent.name: agent.role}, "1+1?")
            summary = ledger.summary()
            self.assertEqual(summary["missing_usage_calls"], 2)
            self.assertEqual(summary["call_count"], len(endpoint.calls))
            self.assertFalse(summary["llm"]["complete"])
            self.assertIsNone(summary["llm"]["total_tokens"])

    async def test_selector_missing_raw_usage_stays_unknown(self):
        from masrubric.usage import TrackedOpenAIChatCompletionClient, usage_scope
        from autogen_core.models import UserMessage

        with FakeModelEndpoint(missing_usage=True) as endpoint:
            client = TrackedOpenAIChatCompletionClient(model="gpt-4o", api_key="local-test",
                                                       base_url=endpoint.url)
            try:
                with usage_scope("selector-unknown") as ledger:
                    await client.create([UserMessage(content="Select an agent to perform task", source="user")])
                summary = ledger.summary()
                self.assertEqual(summary["call_count"], len(endpoint.calls))
                self.assertEqual(summary["missing_usage_calls"], 1)
                self.assertIsNone(summary["llm"]["total_tokens"])
                self.assertFalse(summary["llm"]["complete"])
            finally:
                await client.close()

    async def test_qwen_selector_alias_uses_actual_provider_usage(self):
        from masrubric.usage import TrackedOpenAIChatCompletionClient, usage_scope
        from autogen_core.models import UserMessage

        with FakeModelEndpoint() as endpoint:
            client = TrackedOpenAIChatCompletionClient(model="Qwen/Qwen3-8B", api_key="local-test",
                                                       base_url=endpoint.url)
            try:
                with usage_scope("qwen-selector") as ledger:
                    response = await client.create([UserMessage(content="Select an agent to perform task", source="user")])
                self.assertEqual(response.content, "Participant_1")
                self.assertEqual(endpoint.calls[0]["body"]["model"], "Qwen/Qwen3-8B")
                self.assertEqual(ledger.summary()["llm"]["total_tokens"], endpoint.total())
                self.assertEqual(ledger.summary()["by_stage"]["selector"]["call_count"], 1)
            finally:
                await client.close()

    async def test_http_error_is_recorded_and_propagated(self):
        from masrubric.usage import tracked_create, usage_scope
        from openai import AsyncOpenAI, BadRequestError

        with FakeModelEndpoint(http_error=True) as endpoint:
            async with AsyncOpenAI(base_url=endpoint.url, api_key="local-test", max_retries=0) as client:
                with usage_scope("http-failed") as ledger:
                    with self.assertRaises(BadRequestError):
                        await tracked_create(client.chat.completions.create, stage="reasoning",
                                             model="gpt-4o", messages=[{"role": "user", "content": "1+1?"}])
                summary = ledger.summary()
                self.assertEqual(len(endpoint.calls), 1)
                self.assertEqual(summary["failed_calls"], 1)
                self.assertEqual(summary["call_count"], 1)
                self.assertIsNone(summary["llm"]["total_tokens"])


class UsagePersistenceTests(unittest.TestCase):
    def test_existing_answer_file_is_rejected_without_modification(self):
        from masrubric.run_support import UsageRun

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "results.json"
            original = b'{"old-sample": {"answer": "preserve this"}}\n'
            output.write_bytes(original)
            with self.assertRaises(FileExistsError):
                UsageRun(output, framework="fixed")
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(Path(temp).iterdir()), [output])

    def test_failed_and_missing_results_raise_after_preserving_usage(self):
        from masrubric.run_support import UsageRun, mark_sample_failed
        from masrubric.usage import record_usage

        with tempfile.TemporaryDirectory() as temp:
            for expected_status in ("failed", "no_result"):
                with self.subTest(status=expected_status):
                    run = UsageRun(Path(temp) / f"{expected_status}.json")
                    with run.sample("local"):
                        record_usage("reasoning", {"usage": {"prompt_tokens": 17, "completion_tokens": 3}})
                        if expected_status == "failed":
                            mark_sample_failed(ValueError("deliberate fixture failure"))
                    with self.assertRaisesRegex(RuntimeError, "partial usage was saved"):
                        run.raise_if_failed()
                    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
                    sample = json.loads(run.samples_path.read_text(encoding="utf-8"))
                    self.assertEqual(sample["status"], expected_status)
                    self.assertEqual(summary["samples_failed"], 1)
                    self.assertEqual(summary["samples_completed"], 0)
                    self.assertEqual(summary["token_usage"]["llm"]["total_tokens"], 20)
                    self.assertEqual(len(run.events_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_repeat_runs_and_failed_samples_have_separate_traces(self):
        from masrubric.run_support import UsageRun, attach_usage, mark_sample_failed
        from masrubric.usage import record_usage

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "results.json"
            first = UsageRun(output, framework="fixed")
            with first.sample("same-id"):
                record_usage("reasoning", {"usage": {"prompt_tokens": 7, "completion_tokens": 3}})
                attached = attach_usage({"answer": "2"})
            second = UsageRun(output, framework="fixed")
            with second.sample("same-id"):
                record_usage("reasoning", {"usage": {"prompt_tokens": 17, "completion_tokens": 3}})
                mark_sample_failed(ValueError("private details"))
            one = json.loads(first.summary_path.read_text(encoding="utf-8"))
            two = json.loads(second.summary_path.read_text(encoding="utf-8"))
            self.assertNotEqual(one["run_id"], two["run_id"])
            self.assertEqual(one["token_usage"]["llm"]["total_tokens"], 10)
            self.assertEqual(two["token_usage"]["llm"]["total_tokens"], 20)
            self.assertEqual(one["samples_completed"], 1)
            self.assertEqual(two["samples_failed"], 1)
            self.assertEqual(attached["run_id"], one["run_id"])
            events = [json.loads(line) for line in first.events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["run_id"], one["run_id"])
            self.assertNotIn("private details", second.samples_path.read_text(encoding="utf-8"))
            first.raise_if_failed()
            with self.assertRaises(RuntimeError):
                second.raise_if_failed()


class RunnerFailureTests(unittest.TestCase):
    def test_public_launcher_returns_nonzero_and_preserves_pre_failure_cost(self):
        """A paid first answer followed by HTTP failure must remain in the bill."""
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint(http_error_after=1) as endpoint:
            directory = Path(temp)
            input_file = directory / "input.jsonl"
            input_file.write_text(json.dumps({"id": "local", "question": "What is 1+1?", "answer": "#### 2"})
                                  + "\n", encoding="utf-8")
            environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            for role in ("SELECTOR", "SUPERVISOR", "EMBEDDING"):
                for field in ("URL", "MODEL", "KEY"):
                    environment.pop(f"{role}_{field}", None)
            environment.update(REASONING_URL=endpoint.url, REASONING_MODEL="gpt-4o", REASONING_KEY="local-test")
            command = [sys.executable, str(ROOT / "test" / "run_release_experiment.py"),
                       "--benchmark", "gsm8k", "--method", "fixed_baseline", "--in-file", str(input_file),
                       "--output-dir", str(directory / "output"), "--limit", "1"]
            completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, encoding="utf-8",
                                       errors="replace", capture_output=True, timeout=90)
            self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(len(endpoint.calls), 2)
            summaries = list(directory.rglob("*.usage.summary.json"))
            self.assertEqual(len(summaries), 1, completed.stdout + completed.stderr)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["samples_attempted"], 1)
            self.assertEqual(summary["samples_failed"], 1)
            self.assertEqual(summary["token_usage"]["call_count"], 2)
            self.assertEqual(summary["token_usage"]["failed_calls"], 1)
            self.assertEqual(summary["token_usage"]["llm"]["observed_total_tokens"], endpoint.total())
            self.assertGreater(endpoint.total(), 0)
            self.assertIsNone(summary["token_usage"]["llm"]["total_tokens"])
            events_file = summaries[0].parent / summary["events_file"]
            events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["outcome"] for event in events], ["success", "failed"])
            samples_file = summaries[0].parent / summary["samples_file"]
            self.assertEqual(json.loads(samples_file.read_text(encoding="utf-8"))["status"], "failed")
            self.assertFalse(list(directory.rglob("gsm8k_fixed_baseline.json")))


class RunnerConstructionTests(unittest.TestCase):
    def test_every_benchmark_builds_both_frameworks_from_its_real_cli_parser(self):
        """Catch stale runner args and SDK model-name assumptions without inference."""
        import numpy as np
        from masrubric.teams import FixedDAGTeam
        from autogen_agentchat.teams import SelectorGroupChat

        benchmarks = json.loads((ROOT / "configs" / "release_experiments.json").read_text(encoding="utf-8"))["benchmarks"]
        self.assertEqual(len(benchmarks), 12)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                patch("socket.socket.connect", side_effect=AssertionError("Construction must not connect to a service")):
            for benchmark, spec in benchmarks.items():
                script = ROOT / spec["script"]
                argv = [str(script), "--in_file", str(Path(temp) / "unused.jsonl"),
                        "--out_file", str(Path(temp) / "unused-result.json"),
                        "--metric_pool_file", str(Path(temp) / "unused-metrics.json"),
                        "--embedding_cache_file", str(Path(temp) / "unused-cache.jsonl"),
                        "--baseline_only", "--max_turns", "3", "--framework", "fixed"]
                for component in ("selector", "reasoning", "supervisor", "embedding"):
                    argv.extend([f"--{component}_url", "http://127.0.0.1:1/v1", f"--{component}_key", "local-test",
                                 f"--{component}_model", "Qwen2.5-14B-Instruct"])
                namespace = None
                with self.subTest(benchmark=benchmark, phase="parse"):
                    # Execute the actual CLI block; close its coroutine before
                    # main() runs, then invoke init_team with preloaded resources.
                    with patch.object(sys, "argv", argv), patch("asyncio.run", side_effect=lambda coroutine: coroutine.close()):
                        namespace = runpy.run_path(str(script), run_name="__main__")
                if namespace is None:
                    continue
                for framework, model in (("fixed", "Qwen2.5-14B-Instruct"),
                                         ("dynamic", "gpt-4o"), ("dynamic", "Qwen2.5-14B-Instruct")):
                    with self.subTest(benchmark=benchmark, framework=framework, model=model):
                        namespace["args"].framework = framework
                        namespace["args"].selector_model = model
                        team, final, roles, supervisor = namespace["init_team"]([], np.array([]))
                        self.assertIsInstance(team, FixedDAGTeam if framework == "fixed" else SelectorGroupChat)
                        self.assertEqual(list(roles), [f"Participant_{i}" for i in range(1, 6)])
                        participants = team.participants if framework == "fixed" else team._participants
                        self.assertEqual({agent.name: agent.role for agent in participants}, roles)
                        self.assertTrue(all(agent.supervisor is supervisor for agent in participants))
                        self.assertEqual(final.name, "DecisionMaker")


class RunnerSmokeTests(unittest.TestCase):
    """Full subprocess runs through the existing benchmark CLI entry points."""

    def test_public_launcher_fixed_baseline_needs_only_reasoning_endpoint(self):
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint() as endpoint:
            directory = Path(temp)
            input_file = directory / "input.jsonl"
            input_file.write_text(json.dumps({"id": "local", "question": "What is 1+1?", "answer": "#### 2"})
                                  + "\n", encoding="utf-8")
            environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            for role in ("SELECTOR", "SUPERVISOR", "EMBEDDING"):
                for field in ("URL", "MODEL", "KEY"):
                    environment.pop(f"{role}_{field}", None)
            environment.update(REASONING_URL=endpoint.url, REASONING_MODEL="gpt-4o", REASONING_KEY="local-test")
            command = [sys.executable, str(ROOT / "test" / "run_release_experiment.py"),
                       "--benchmark", "gsm8k", "--method", "fixed_baseline", "--in-file", str(input_file),
                       "--output-dir", str(directory / "output"), "--limit", "1",
                       "--metric-pool-file", str(directory / "intentionally-absent-metrics.json"),
                       "--embedding-cache-file", str(directory / "intentionally-absent-cache.jsonl")]
            completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, encoding="utf-8",
                                       errors="replace", capture_output=True, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout[-12000:] + completed.stderr[-6000:])
            outputs = list(directory.rglob("gsm8k_fixed_baseline.json"))
            self.assertEqual(len(outputs), 1, completed.stdout)
            record = next(iter(json.loads(outputs[0].read_text(encoding="utf-8")).values()))
            self.assertTrue(record["is_correct"])
            self.assertEqual(record["framework"], "fixed")
            self.assert_accounted(record, endpoint)

    def run_fixture(self, benchmark, endpoint, directory, *, framework="fixed", audited=False):
        directory = Path(directory)
        input_file, metrics_file = directory / "input.jsonl", directory / "metrics.json"
        cache_file, output_file = directory / "cache.jsonl", directory / "results.json"
        if benchmark == "humaneval":
            fixture = {"task_id": "local/add", "prompt": "def add(a, b):\n    '''Return a + b.'''\n",
                       "test": "def check(candidate):\n    assert candidate(1, 1) == 2\n",
                       "entry_point": "add", "canonical_solution": "    return a + b\n"}
        elif benchmark == "math500":
            fixture = {"id": "local-1", "problem": "What is 1+1?", "solution": "\\boxed{2}", "answer": "2"}
        else:
            fixture = {"id": "local-1", "question": "What is 1+1?", "answer": "Add the numbers. #### 2"}
        input_file.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
        metrics_file.write_text(json.dumps(METRICS), encoding="utf-8")
        cache_file.write_text("".join(json.dumps({"name": m["name"], "vector": [1.0, 0.0]}) + "\n"
                                      for m in METRICS), encoding="utf-8")
        command = [sys.executable, str(ROOT / "test" / "experiments" / benchmark / f"run_{benchmark}.py"),
                   "--in_file", str(input_file), "--out_file", str(output_file),
                   "--metric_pool_file", str(metrics_file), "--embedding_cache_file", str(cache_file),
                   "--max_turns", "3", "--limit", "1", "--framework", framework,
                   "--retries_times", "1"]
        for component in ["selector", "reasoning", "supervisor", "embedding"]:
            command.extend([f"--{component}_url", endpoint.url, f"--{component}_key", "local-test",
                            f"--{component}_model", "embedding-local" if component == "embedding" else "gpt-4o"])
        if audited:
            command.append("--use_simple_audit")
        else:
            command.append("--baseline_only")
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        completed = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                   encoding="utf-8", errors="replace", capture_output=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout[-12000:] + completed.stderr[-6000:])
        self.assertTrue(output_file.exists(), completed.stdout[-12000:] + completed.stderr[-6000:])
        records = json.loads(output_file.read_text(encoding="utf-8"))
        self.assertEqual(len(records), 1, str(records))
        record = next(iter(records.values()))
        self.assertIn("token_usage", record, str(record))
        return record, list(directory.glob("*.usage.summary.json")), completed

    def assert_accounted(self, record, endpoint):
        summary = record["token_usage"]
        self.assertEqual(summary["call_count"], len(endpoint.calls))
        self.assertEqual(summary["llm"]["total_tokens"], endpoint.total())
        self.assertEqual(summary["embedding"]["total_tokens"], endpoint.total("embedding"))
        self.assertTrue(summary["llm"]["complete"])

    def test_gsm8k_fixed_baseline(self):
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint() as endpoint:
            record, summaries, _ = self.run_fixture("gsm8k", endpoint, temp)
            self.assertTrue(record["is_correct"])
            self.assertFalse(any(c["stage"] == "selector" for c in endpoint.calls))
            self.assert_accounted(record, endpoint)
            self.assertTrue(summaries)

    def test_gsm8k_dynamic_baseline_counts_selector(self):
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint() as endpoint:
            record, _, _ = self.run_fixture("gsm8k", endpoint, temp, framework="dynamic")
            self.assertTrue(record["is_correct"])
            self.assertTrue(any(c["stage"] == "selector" for c in endpoint.calls))
            self.assert_accounted(record, endpoint)

    def test_gsm8k_fixed_audited_retries_and_accounts_all_nodes(self):
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint(fail_audits=1) as endpoint:
            record, _, _ = self.run_fixture("gsm8k", endpoint, temp, audited=True)
            self.assertTrue(record["is_correct"])
            self.assertEqual(endpoint.audit_count, 6)
            self.assertFalse(any(c["stage"] == "selector" for c in endpoint.calls))
            self.assertNotIn("fallback", record["token_usage"]["by_phase"])
            self.assert_accounted(record, endpoint)

    def test_math500_fixed_baseline(self):
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint() as endpoint:
            record, _, _ = self.run_fixture("math500", endpoint, temp)
            self.assertTrue(record["is_correct"])
            self.assert_accounted(record, endpoint)

    def test_humaneval_fixed_baseline_executes_generated_code(self):
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint(code=True) as endpoint:
            record, _, _ = self.run_fixture("humaneval", endpoint, temp)
            self.assertTrue(record["solved"], record)
            self.assert_accounted(record, endpoint)

    def test_gsm8k_fallback_preserves_both_phases_cost(self):
        with tempfile.TemporaryDirectory() as temp, FakeModelEndpoint(fail_audits=100) as endpoint:
            record, _, _ = self.run_fixture("gsm8k", endpoint, temp, framework="dynamic", audited=True)
            self.assertTrue(record["is_correct"])
            self.assertGreater(endpoint.audit_count, 0)
            self.assertIn("fallback", record["token_usage"]["by_phase"])
            self.assertGreater(record["token_usage"]["by_phase"]["primary"]["llm"]["call_count"], 0)
            self.assertGreater(record["token_usage"]["by_phase"]["fallback"]["llm"]["call_count"], 1)
            self.assert_accounted(record, endpoint)


if __name__ == "__main__":
    unittest.main()
