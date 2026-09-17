"""Contract tests for request-level accounting, without an external model service.

Run with: python -m unittest discover -s tests -p 'test_*.py'
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))

from AgentDropout.usage import (  # noqa: E402
    current_usage,
    record_usage,
    set_usage_phase,
    tracked_create,
    tracked_embedding_create,
    usage_scope,
)


def completion(prompt=11, output=7):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=output,
                              total_tokens=prompt + output),
    )


class UsageTrackingTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_successful_request_is_counted_before_caller_parses_it(self):
        """A rejected answer or JSON parse error does not refund its model call."""
        responses = iter([completion(11, 3), completion(13, 5), completion(17, 7)])

        async def create(**kwargs):
            return next(responses)

        with usage_scope("retried-answer") as ledger:
            for attempt in range(3):
                await tracked_create(create, stage="participant", source="A",
                                     model="local", metadata={"attempt": attempt + 1})
        summary = ledger.summary()
        self.assertEqual(summary["call_count"], 3)
        self.assertEqual(summary["llm"]["prompt_tokens"], 41)
        self.assertEqual(summary["llm"]["completion_tokens"], 15)
        self.assertEqual(summary["llm"]["total_tokens"], 56)
        self.assertTrue(summary["llm"]["complete"])

    async def test_missing_usage_is_unknown_not_zero(self):
        with usage_scope("missing") as ledger:
            record_usage("participant", completion(5, 3))
            record_usage("final", SimpleNamespace(usage=None))
        summary = ledger.summary()
        self.assertEqual(summary["call_count"], 2)
        self.assertEqual(summary["missing_usage_calls"], 1)
        self.assertFalse(summary["llm"]["complete"])
        self.assertIsNone(summary["llm"]["total_tokens"])
        self.assertEqual(summary["llm"]["observed_total_tokens"], 8)

    async def test_partial_usage_keeps_only_observed_values(self):
        partial = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=9))
        with usage_scope("partial") as ledger:
            record_usage("participant", partial)
        summary = ledger.summary()
        self.assertFalse(summary["llm"]["complete"])
        self.assertIsNone(summary["llm"]["total_tokens"])
        self.assertEqual(summary["llm"]["observed_prompt_tokens"], 9)

    async def test_failed_request_propagates_and_marks_cost_incomplete(self):
        async def fail(**kwargs):
            raise RuntimeError("secret-key-and-private-endpoint")

        with usage_scope("failed") as ledger:
            with self.assertRaisesRegex(RuntimeError, "secret-key"):
                await tracked_create(fail, stage="supervisor.audit", model="local")
        summary = ledger.summary()
        self.assertEqual(summary["failed_calls"], 1)
        self.assertFalse(summary["llm"]["complete"])
        self.assertIsNone(summary["llm"]["total_tokens"])
        self.assertNotIn("secret-key", str(ledger.events))
        self.assertIn("RuntimeError", str(ledger.events))

    async def test_fallback_retains_initial_calls(self):
        with usage_scope("fallback") as ledger:
            record_usage("participant", completion(10, 2))
            record_usage("supervisor.audit", completion(8, 3))
            set_usage_phase("fallback")
            record_usage("participant", completion(12, 4))
            record_usage("final", completion(14, 5))
        summary = ledger.summary()
        self.assertEqual(summary["call_count"], 4)
        self.assertEqual(summary["llm"]["total_tokens"], 58)
        self.assertEqual(summary["by_phase"]["fallback"]["llm"]["call_count"], 2)

    async def test_embedding_usage_is_separate_from_llm_usage(self):
        def embed(**kwargs):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=23, total_tokens=23))

        with usage_scope("embedding") as ledger:
            record_usage("participant", completion(10, 4))
            tracked_embedding_create(embed, stage="embedding.query", model="embed-local", input=["q"])
        summary = ledger.summary()
        self.assertEqual(summary["llm"]["total_tokens"], 14)
        self.assertEqual(summary["embedding"]["total_tokens"], 23)
        self.assertEqual(summary["embedding"]["completion_tokens"], 0)
        self.assertEqual(summary["call_count"], 2)

    async def test_concurrent_samples_and_phases_do_not_leak(self):
        barrier = asyncio.Event()

        async def sample(sample_id, prompt):
            with usage_scope(sample_id) as ledger:
                record_usage("participant", completion(prompt, 2))
                await barrier.wait()
                if sample_id == "a":
                    set_usage_phase("fallback")
                await asyncio.sleep(0)
                record_usage("final", completion(prompt, 3))
                return ledger.summary()

        first = asyncio.create_task(sample("a", 11))
        second = asyncio.create_task(sample("b", 101))
        await asyncio.sleep(0)
        barrier.set()
        a, b = await asyncio.gather(first, second)
        self.assertEqual(a["sample_id"], "a")
        self.assertEqual(b["sample_id"], "b")
        self.assertEqual(a["llm"]["total_tokens"], 27)
        self.assertEqual(b["llm"]["total_tokens"], 207)
        self.assertIn("fallback", a["by_phase"])
        self.assertNotIn("fallback", b["by_phase"])
        self.assertIsNone(current_usage())

    async def test_nested_scope_restores_outer_and_exception_cleans_context(self):
        with usage_scope("outer") as outer:
            record_usage("participant", completion(1, 1))
            with self.assertRaises(ValueError):
                with usage_scope("inner") as inner:
                    record_usage("participant", completion(20, 2))
                    raise ValueError("bad response")
            self.assertIs(current_usage(), outer)
            record_usage("final", completion(2, 2))
        self.assertEqual(outer.summary()["llm"]["total_tokens"], 6)
        self.assertEqual(inner.summary()["llm"]["total_tokens"], 22)
        self.assertIsNone(current_usage())

    async def test_existing_actor_task_sees_fallback_phase_after_reset(self):
        ready = asyncio.Event()
        resume = asyncio.Event()

        async def actor():
            ready.set()
            await resume.wait()
            record_usage("reasoning", completion(9, 2))

        with usage_scope("actor-fallback") as ledger:
            task = asyncio.create_task(actor())
            await ready.wait()
            record_usage("reasoning", completion(3, 2))
            set_usage_phase("fallback")
            resume.set()
            await task
        self.assertEqual(ledger.summary()["by_phase"]["fallback"]["llm"]["total_tokens"], 11)
        self.assertEqual(ledger.summary()["by_phase"]["primary"]["llm"]["total_tokens"], 5)

    async def test_unscoped_client_calls_remain_usable(self):
        async def create(**kwargs):
            return completion()

        response = await tracked_create(create, stage="participant", model="local")
        self.assertEqual(response.usage.total_tokens, 18)
        self.assertIsNone(current_usage())


if __name__ == "__main__":
    unittest.main()
