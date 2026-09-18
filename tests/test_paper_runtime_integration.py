"""Run recovered math/code entrypoints through real, isolated child processes.

Only a local HTTP fixture is contacted. Synthetic datasets have the declared
benchmark size to exercise preflight, while --limit 1 executes one fixture row.
These tests verify wiring and recorded usage, not historical benchmark scores.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from test_release_integration import FakeModelEndpoint, finding


ROOT = Path(__file__).resolve().parents[1]


class PaperModelEndpoint(FakeModelEndpoint):
    """Speak the recovered runtime's selected_metrics/judgements schemas."""

    def reply(self, path, body):
        response = super().reply(path, body)
        if path.endswith("/embeddings"):
            return response
        prompt = "\n".join(str(message.get("content", "")) for message in body.get("messages", []))
        stage = None
        if "Select an agent to perform task" in prompt:
            # The third member is the programming expert in both recovered
            # teams, and the mathematical variant turns it into a verifier.
            content, stage = "Participant_3", "selector"
        elif "compact JSON match profile for code metric retrieval" in prompt:
            content = json.dumps({"interface_shape": "function_api", "io_contract": "return_value",
                                  "match_hint": "The add function returns the sum of its two arguments."})
            stage = "match_profile"
        elif "Lead Logic Auditor" in prompt and "selected_metrics" in prompt:
            names = re.findall(r"^- Metric: (.+)$", prompt, flags=re.MULTILINE)
            content = json.dumps({"summary": "Verify the addition.", "selected_metrics": names[:5]})
            stage = "rerank"
        elif "Evaluate the agent output against every metric" in prompt:
            names = re.findall(r"^\d+\. Metric Name: (.+)$", prompt, flags=re.MULTILINE)
            content = json.dumps({"judgements": [finding(metric=name) for name in names]})
            stage = "batch_audit"
        if stage:
            response["choices"][0]["message"]["content"] = content
            self.calls[-1]["stage"] = stage
        return response


def metric(name, shape="any", io_contract="any"):
    return {"name": name, "detailed_definition": "Verify the addition and required output.",
            "evaluator_prompt": {"trigger_condition": "Arithmetic addition", "risk_alert": "Check the result."},
            "match_profile": {"interface_shape": shape, "io_contract": io_contract,
                              "match_hint": "Verify the interface and returned numeric value."}}


class PaperFinalTemperatureTests(unittest.TestCase):
    def test_final_requests_use_historical_temperatures_and_code4_routing(self):
        # Import the recovered runtime in a separate interpreter: the other
        # integration tests also import the public package named AgentDropout.
        script = r"""
import asyncio
import inspect
import json
from pathlib import Path
import sys

runtime = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(runtime), str(runtime / "scripts")]
from AgentDropout.agents import AgentRegistry
from v2_code_common import ensure_task_name

async def main():
    resolved = []
    for benchmark in ("mbpp", "humaneval", "codecontest", "livecode", "gsm8k"):
        if benchmark == "gsm8k":
            agent_name, domain = "FinalRefer", "gsm8k"
        else:
            spec = ensure_task_name(benchmark)
            agent_name, domain = spec.decision_agent_name, spec.domain_name
        decision = AgentRegistry.get(
            agent_name, name="DecisionMaker", model="gpt-4o",
            api_key="local-test", base_url=sys.argv[2], domain=domain,
        )
        try:
            answer = await decision.run_decision([], {}, "What is 1+1?")
            resolved.append({
                "benchmark": benchmark,
                "class": type(decision).__name__,
                "module": str(Path(inspect.getfile(type(decision))).resolve()),
                "source": answer.source,
            })
        finally:
            await decision._model_client.close()
    print(json.dumps(resolved))

asyncio.run(main())
"""
        environment = {
            name: value for name, value in os.environ.items()
            if not name.startswith("AGENTDROPOUT_")
        }
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                           NO_PROXY="127.0.0.1,localhost")
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            environment.pop(name, None)
        runtime = ROOT / "paper/runtime"
        expected = [
            ("mbpp", "FinalWriteCodeMBPP", 0.0),
            ("humaneval", "FinalWriteCode", 0.0),
            ("codecontest", "FinalWriteCode", 0.0),
            ("livecode", "FinalWriteCode", 0.0),
            ("gsm8k", "FinalRefer", 0.7),
        ]
        config = json.loads((ROOT / "configs/paper_main.json").read_text(encoding="utf-8"))
        with FakeModelEndpoint() as endpoint:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script, str(runtime), endpoint.url],
                cwd=ROOT, env=environment, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            resolved = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(len(resolved), len(expected))
            self.assertEqual(len(endpoint.calls), len(expected))
            for (benchmark, agent_name, temperature), decision, call in zip(expected, resolved, endpoint.calls):
                with self.subTest(benchmark=benchmark):
                    self.assertEqual(decision["benchmark"], benchmark)
                    self.assertEqual(decision["class"], agent_name)
                    self.assertEqual(decision["source"], "DecisionMaker")
                    self.assertEqual(Path(decision["module"]),
                                     (runtime / "AgentDropout/agents/final_decision.py").resolve())
                    self.assertTrue(call["path"].endswith("/chat/completions"))
                    self.assertEqual(call["body"]["model"], "gpt-4o")
                    # This is the JSON received over HTTP, after the real SDK
                    # has applied request arguments, rather than an AST check.
                    self.assertEqual(call["body"]["temperature"], temperature)
                    self.assertEqual(config["benchmarks"][benchmark]["final_decision"],
                                     {"class": decision["class"], "temperature": call["body"]["temperature"]})


class PaperRuntimeIntegrationTests(unittest.TestCase):
    def run_fixture(self, benchmark, endpoint, directory):
        config = json.loads((ROOT / "configs/paper_main.json").read_text(encoding="utf-8"))
        is_code = benchmark == "mbpp"
        suite = "code_8b" if is_code else "math_8b"
        count = config["benchmarks"][benchmark]["total"]
        if is_code:
            rows = [{"task_id": f"fixture-{index}", "text": "Write add(a, b) returning the sum of a and b.",
                     "code": "def add(a, b):\n    return a + b\n",
                     "test_list": ["assert add(1, 1) == 2", "assert add(-2, 3) == 1"]}
                    for index in range(count)]
            metrics = [metric("conflicting_stdio", "stdin_stdout", "stdout")]
            metrics += [metric(f"function_check_{index}", "function_api", "return_value") for index in range(3)]
            # The incompatible metric has the best similarity. Its absence
            # from every audit demonstrates that the hard filter actually ran.
            vectors = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
        else:
            rows = [{"id": f"fixture-{index}", "question": "What is 1+1?", "answer": "Add the numbers. #### 2"}
                    for index in range(count)]
            metrics = [metric(f"arithmetic_check_{index}") for index in range(5)]
            vectors = [[1.0, 0.0] for _ in metrics]

        dataset = directory / "input.jsonl"
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        pool, cache = directory / "metrics.json", directory / "embeddings.jsonl"
        pool.write_text(json.dumps(metrics), encoding="utf-8")
        cache.write_text("".join(json.dumps({"name": item["name"], "vector": vector}) + "\n"
                                 for item, vector in zip(metrics, vectors)), encoding="utf-8")
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                           PYTHONPATH=str(ROOT / "test"), NO_PROXY="127.0.0.1,localhost")
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            environment.pop(name, None)
        # Deliberately contaminate the caller's settings. The paper entrypoint
        # must lock its main-method settings before importing the child runtime.
        environment.update(AGENTDROPOUT_MATH_TEAM_VARIANT="disabled",
                           AGENTDROPOUT_PROFILE_AWARE_RETRIEVAL="0",
                           AGENTDROPOUT_BATCH_AUDIT_METRICS="0",
                           AGENTDROPOUT_EXACT_SELECT_Q="1", AGENTDROPOUT_CHEAP_PRECHECK="1",
                           AGENTDROPOUT_RANDOM_K_MIN="1", AGENTDROPOUT_RANDOM_K_MAX="1")
        for role in ("SELECTOR", "REASONING", "SUPERVISOR", "EMBEDDING"):
            environment[f"{role}_URL"] = endpoint.url
            environment[f"{role}_MODEL"] = "embedding-local" if role == "EMBEDDING" else "gpt-4o"
            environment[f"{role}_KEY"] = "local-test"
        command = [sys.executable, str(ROOT / "test/run_paper_main.py"), "--suite", suite,
                   "--benchmark", benchmark, "--in-file", str(dataset), "--metric-pool-file", str(pool),
                   "--embedding-cache-file", str(cache), "--output-dir", str(directory / "output"),
                   "--limit", "1", "--timeout", "90", "--allow-model-override"]
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=120)
        logs = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                         for path in directory.rglob("*.log"))
        self.assertEqual(completed.returncode, 0, (completed.stdout + completed.stderr + logs)[-24000:])
        manifests = list(directory.rglob("manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        task_dir = manifests[0].parent
        summary = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["complete"], summary)
        self.assertEqual(summary["completed_samples"], 1)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["scope"], "smoke_subset")
        self.assertEqual(Path(manifest["runtime"]).resolve(), (ROOT / "paper/runtime").resolve())
        fixed = manifest["fixed_environment"]
        self.assertEqual(fixed["AGENTDROPOUT_BATCH_AUDIT_METRICS"], "1")
        self.assertEqual(fixed["AGENTDROPOUT_EXACT_SELECT_Q"], "0")
        self.assertEqual(fixed["AGENTDROPOUT_CHEAP_PRECHECK"], "0")
        events = [json.loads(line) for line in (task_dir / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(events), len(endpoint.calls))
        self.assertEqual(sum(event["total_tokens"] for event in events if event["kind"] == "chat_completion"),
                         endpoint.total())
        self.assertEqual(sum(event["total_tokens"] for event in events if event["kind"] == "embedding"),
                         endpoint.total("embedding"))
        self.assertTrue({"selector", "reasoning", "final_decision", "supervisor_audit_batch"}
                        .issubset({event["stage"] for event in events}))
        return manifest, events, logs

    def test_gsm8k_uses_verifier_and_rerank_in_the_isolated_runtime(self):
        with tempfile.TemporaryDirectory() as temp, PaperModelEndpoint() as endpoint:
            manifest, events, _ = self.run_fixture("gsm8k", endpoint, Path(temp))
            self.assertEqual(manifest["fixed_environment"]["AGENTDROPOUT_MATH_TEAM_VARIANT"], "verifier")
            reasoning_prompts = ["\n".join(message.get("content", "") for message in call["body"]["messages"])
                                 for call in endpoint.calls if call["stage"] == "reasoning"]
            self.assertTrue(any("computational verifier" in prompt for prompt in reasoning_prompts))
            self.assertTrue(any(event["stage"] == "supervisor_rerank" for event in events))
            audits = [call for call in endpoint.calls if call["stage"] == "batch_audit"]
            self.assertTrue(audits)
            for call in audits:
                prompt = call["body"]["messages"][-1]["content"]
                self.assertEqual(len(re.findall(r"^\d+\. Metric Name:", prompt, re.MULTILINE)), 5)

    def test_mbpp_executes_code_and_hard_filters_conflicting_metrics(self):
        with tempfile.TemporaryDirectory() as temp, PaperModelEndpoint(code=True) as endpoint:
            manifest, events, logs = self.run_fixture("mbpp", endpoint, Path(temp))
            self.assertEqual(manifest["fixed_environment"]["AGENTDROPOUT_PROFILE_AWARE_RETRIEVAL"], "1")
            self.assertTrue(any(event["stage"] == "supervisor_match_profile" for event in events))
            self.assertTrue(any(event["stage"] == "embedding_query_profileaware" for event in events))
            self.assertFalse(any(event["stage"] == "supervisor_rerank" for event in events))
            self.assertIn("Hard-filtered conflicting metrics: 1", logs)
            audits = [call for call in endpoint.calls if call["stage"] == "batch_audit"]
            self.assertTrue(audits)
            for call in audits:
                prompt = call["body"]["messages"][-1]["content"]
                names = re.findall(r"^\d+\. Metric Name: (.+)$", prompt, re.MULTILINE)
                self.assertEqual(set(names), {f"function_check_{index}" for index in range(3)})
                self.assertNotIn("conflicting_stdio", prompt)


if __name__ == "__main__":
    unittest.main()
