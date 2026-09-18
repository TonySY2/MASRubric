"""Regression coverage for launch inputs and bounded participant failures.

The launcher tests use only the standard library. Participant and runner tests
use the pinned inference dependencies, but never call an external model service.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))

import run_release_experiment as launcher


class LauncherPoolOverrideTests(unittest.TestCase):
    def arguments(self, **overrides):
        values = dict(benchmark="gsm8k", method="adv2_math_main", framework=None,
                      fixed_rounds=None, limit=None, pool=None, in_file=None,
                      metric_pool_file=None, embedding_cache_file=None,
                      output_dir="test/results_release", model_profile=None,
                      dry_run=True, max_turns=None)
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_pool_environment_overrides_apply_to_every_public_pool(self):
        config = launcher.load_config()
        environment = {"AGENTDROPOUT_METRIC_POOL_FILE": "external/metrics.json",
                       "AGENTDROPOUT_EMBEDDING_CACHE_FILE": "external/cache.jsonl"}
        with patch.dict(os.environ, environment, clear=True):
            for pool in config["metric_pools"]:
                with self.subTest(pool=pool):
                    command = launcher.build_command(self.arguments(pool=pool), config)
                    self.assertEqual(command[command.index("--metric_pool_file") + 1],
                                     str(ROOT / "external/metrics.json"))
                    self.assertEqual(command[command.index("--embedding_cache_file") + 1],
                                     str(ROOT / "external/cache.jsonl"))

    def test_explicit_paths_take_precedence_over_environment(self):
        environment = {"AGENTDROPOUT_METRIC_POOL_FILE": "ignored/metrics.json",
                       "AGENTDROPOUT_EMBEDDING_CACHE_FILE": "ignored/cache.jsonl"}
        with patch.dict(os.environ, environment, clear=True):
            command = launcher.build_command(
                self.arguments(metric_pool_file="chosen/metrics.json", embedding_cache_file="chosen/cache.jsonl"),
                launcher.load_config())
        self.assertEqual(command[command.index("--metric_pool_file") + 1], str(ROOT / "chosen/metrics.json"))
        self.assertEqual(command[command.index("--embedding_cache_file") + 1], str(ROOT / "chosen/cache.jsonl"))

    def test_unset_overrides_keep_the_selected_pool_defaults(self):
        config = launcher.load_config()
        with patch.dict(os.environ, {}, clear=True):
            for pool, spec in config["metric_pools"].items():
                with self.subTest(pool=pool):
                    command = launcher.build_command(self.arguments(pool=pool), config)
                    for field in ("metric_pool_file", "embedding_cache_file"):
                        self.assertEqual(command[command.index("--" + field) + 1],
                                         str(ROOT / spec[field]))

    def test_top3_ablation_requires_all_three_indicators_to_pass(self):
        with patch.dict(os.environ, {}, clear=True):
            command = launcher.build_command(self.arguments(method="adv2_math_top3"), launcher.load_config())
        self.assertEqual(command[command.index("--retrieve_p") + 1], "20")
        self.assertEqual(command[command.index("--select_q") + 1], "3")
        self.assertEqual(command[command.index("--pass_rate") + 1], "1.0")


class ParticipantRetryTests(unittest.IsolatedAsyncioTestCase):
    def make_agent(self, benchmark, retries):
        from AgentDropout.agents import AgentRegistry

        kind = "CodeWriting" if benchmark in {"mbpp", "humaneval", "codecontest", "livecode"} else "MathSolver"
        return AgentRegistry.get(agent_name=f"{kind}_{benchmark}", domain=benchmark,
                                 name="Participant_1", model="local-fixture", api_key="local-test",
                                 base_url="http://127.0.0.1:1/v1", message_history=[], reflection_time=retries)

    async def test_persistent_connection_failures_exhaust_each_agents_budget(self):
        import httpx
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        from openai import APIConnectionError
        from AgentDropout.usage import usage_scope

        request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
        for benchmark in launcher.load_config()["benchmarks"]:
            for retries in (0, 2):
                with self.subTest(benchmark=benchmark, retries=retries):
                    agent = self.make_agent(benchmark, retries)
                    error = APIConnectionError(request=request)
                    # A sentinel fails quickly if a regression exceeds the budget.
                    create = AsyncMock(side_effect=[error] * (retries + 1) +
                                       [AssertionError("Participant exceeded its retry budget")])
                    try:
                        with patch.object(agent._model_client.chat.completions, "create", create), \
                                patch("asyncio.sleep", new_callable=AsyncMock) as sleep, \
                                contextlib.redirect_stdout(io.StringIO()), \
                                usage_scope("connection-failure") as ledger:
                            with self.assertRaises(APIConnectionError):
                                await agent.on_messages([TextMessage(content="What is 1+1?", source="user")],
                                                        CancellationToken())
                        self.assertEqual(create.await_count, retries + 1)
                        self.assertEqual(sleep.await_count, retries)
                        self.assertEqual(ledger.summary()["failed_calls"], retries + 1)
                        self.assertEqual([event["metadata"]["agent_attempt"] for event in ledger.events],
                                         list(range(1, retries + 2)))
                    finally:
                        await agent._model_client.close()

    async def test_transient_timeout_recovers_on_the_next_attempt(self):
        import httpx
        from autogen_agentchat.messages import TextMessage
        from autogen_core import CancellationToken
        from openai import APITimeoutError
        from AgentDropout.usage import usage_scope

        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(model_dump=Mock(return_value={"content": "2"})))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14))
        request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
        for benchmark in launcher.load_config()["benchmarks"]:
            with self.subTest(benchmark=benchmark):
                agent = self.make_agent(benchmark, 1)
                create = AsyncMock(side_effect=[APITimeoutError(request=request), completion])
                try:
                    with patch.object(agent._model_client.chat.completions, "create", create), \
                            patch("asyncio.sleep", new_callable=AsyncMock), \
                            contextlib.redirect_stdout(io.StringIO()), usage_scope("transient-failure") as ledger:
                        response = await agent.on_messages([TextMessage(content="What is 1+1?", source="user")],
                                                           CancellationToken())
                    self.assertEqual(response.chat_message.content, "2")
                    self.assertEqual(create.await_count, 2)
                    self.assertEqual(ledger.summary()["failed_calls"], 1)
                    self.assertEqual(ledger.summary()["llm"]["observed_total_tokens"], 14)
                    self.assertEqual([event["metadata"]["agent_attempt"] for event in ledger.events], [1, 2])
                finally:
                    await agent._model_client.close()


class RunnerInputFailureTests(unittest.TestCase):
    def test_every_runner_exits_nonzero_for_missing_or_malformed_input(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            malformed = directory / "malformed.jsonl"
            malformed.write_text("{invalid json\n", encoding="utf-8")
            for benchmark, spec in launcher.load_config()["benchmarks"].items():
                for source, error_name in ((directory / "missing.jsonl", "FileNotFoundError"),
                                           (malformed, "JSONDecodeError")):
                    with self.subTest(benchmark=benchmark, input=source.name):
                        output = directory / f"{benchmark}-{source.stem}.json"
                        command = [sys.executable, str(ROOT / spec["script"]), "--in_file", str(source),
                                   "--out_file", str(output), "--baseline_only", "--framework", "fixed",
                                   "--embedding_url", "http://127.0.0.1:1/v1", "--embedding_model", "unused",
                                   "--metric_pool_file", str(directory / "unused-metrics.json"),
                                   "--embedding_cache_file", str(directory / "unused-cache.jsonl")]
                        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                                   encoding="utf-8", errors="replace", timeout=45,
                                                   env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))
                        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                        # An import/setup failure must not masquerade as passing this regression.
                        self.assertIn(error_name, completed.stderr)
                        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
