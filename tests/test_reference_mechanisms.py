"""Mechanism controls exercise the recovered agents; all responses are fixtures."""
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mechanism_launcher", ROOT / "test/run_paper_main.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class MechanismLauncherTests(unittest.TestCase):
    def plan(self, *flags, suite="math_8b", benchmark="gsm8k"):
        args = launcher.parser().parse_args(["--suite", suite, "--benchmark", benchmark, *flags])
        return launcher.build_plan(args, launcher.read_json(launcher.CONFIG), benchmark, Path("fixture-output"))

    def test_revise_only_keeps_rubric_threshold_and_revision_budget(self):
        full, full_env, _, _ = self.plan()
        revised, env, manifest, _ = self.plan("--variant-name", "revise_only", "--withholding-policy", "broadcast")
        self.assertEqual(full, revised)
        self.assertEqual(full_env["MASRUBRIC_WITHHOLDING_POLICY"], "withhold")
        self.assertEqual(env["MASRUBRIC_WITHHOLDING_POLICY"], "broadcast")
        self.assertEqual(manifest["parameter_overrides"], {"withholding_policy": "broadcast"})

    def test_code_zero_revision_is_explicit_and_requires_variant(self):
        with self.assertRaisesRegex(ValueError, "variant-name"):
            self.plan("--withholding-policy", "broadcast")
        _, env, _, _ = self.plan("--variant-name", "withhold_only", "--retry-budget", "0", suite="code_8b", benchmark="mbpp")
        self.assertEqual(env["MASRUBRIC_CODE_RETRY_BUDGET"], "0")


class ActualReferenceMechanismTests(unittest.TestCase):
    def test_tracing_preserves_tolerated_malformed_rerank_selection(self):
        script = r'''
import asyncio, json, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
sys.path.insert(0, sys.argv[1])
import numpy as np
from autogen_agentchat.messages import TextMessage
from masrubric.agents.supervisor_reasoning_pick_metric import Supervisor

async def main():
    metric = {"name": "sum", "detailed_definition": "Check addition", "evaluator_prompt": {"trigger_condition": "sum", "risk_alert": "check sum"}}
    supervisor = Supervisor(domain='math500', model='fixture', api_key='EMPTY', base_url='http://127.0.0.1:1/v1',
        embedding_api_key='EMPTY', embedding_api_base='http://127.0.0.1:1/v1',
        preloaded_metrics=[metric], preloaded_embeddings=np.array([[1., 0.]]))
    supervisor.embedding_client.embeddings.create = lambda **kw: SimpleNamespace(
        model='fixture', usage=None, data=[SimpleNamespace(embedding=[1., 0.])])
    def reply(value):
        return SimpleNamespace(model='fixture', usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])
    for selected in (None, 7, {'invalid': 'shape'}, [{'unexpected': 'object'}], [['nested']]):
        supervisor._model_client.chat.completions.create = AsyncMock(side_effect=[
            reply({'problem_scenario': ['addition'], 'agent_action': ['sum']}),
            reply({'selected_metrics': selected})])
        message = TextMessage(source='solver', content='1+1=2')
        passed, judgments, feedback, metrics = await supervisor.judge('1+1?', message, 1)
        assert metrics == supervisor._apply_select_q_policy([metric], selected) == []
        assert passed and judgments == [] and feedback is None
        assert supervisor._selection_trace['fallback'] is False
        assert supervisor._selection_trace['selected_names_returned'] == selected
        supervisor.update_scoreboard_with_results(message, judgments)
        assert supervisor.scoreboard[message.id]['audit_attempts'][0]['empty_cause'] == 'no_selected_criteria'
    await supervisor._model_client.close()
    supervisor.embedding_client.close()
asyncio.run(main())
'''
        environment = {k: v for k, v in os.environ.items() if not k.startswith("MASRUBRIC_")}
        environment["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT / "paper/runtime")],
                                env=environment, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_full_launcher_preserves_fallback_trace_and_broadcast_avoids_pruning(self):
        import test_paper_runtime_integration as fixtures
        from test_release_integration import finding

        class FlawedEndpoint(fixtures.PaperModelEndpoint):
            def reply(self, path, body):
                response = super().reply(path, body)
                if self.calls[-1]["stage"] == "batch_audit":
                    prompt = body["messages"][-1]["content"]
                    names = re.findall(r"^\d+\. Metric Name: (.+)$", prompt, flags=re.MULTILINE)
                    response["choices"][0]["message"]["content"] = json.dumps(
                        {"judgements": [finding(flawed=True, metric=name) for name in names]})
                return response

        helper = fixtures.PaperRuntimeIntegrationTests()
        for policy in ("withhold", "broadcast"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory, FlawedEndpoint() as endpoint:
                flags = ("--variant-name", "mechanism_fixture", "--withholding-policy", policy)
                manifest, _, logs = helper.run_fixture("gsm8k", endpoint, Path(directory), flags)
                path = Path(manifest["result_file"]).parent / "mechanism.jsonl"
                events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                finalized = [e for e in events if e["event_type"] == "message_finalized"]
                audited = [e for e in finalized if e["audit_attempts"]]
                self.assertEqual(len(audited), 6)
                self.assertTrue(all(len(e["audit_attempts"]) == 4 for e in audited))
                self.assertTrue(all(e["is_pruned"] == (policy == "withhold") for e in audited))
                if policy == "withhold":
                    self.assertIn("Fallback Triggered", logs)
                    self.assertEqual(len(finalized), 12)
                    self.assertEqual(len({e["phase"] for e in finalized}), 2)
                else:
                    self.assertNotIn("Fallback Triggered", logs)
                    self.assertEqual(len(finalized), 6)

    def test_agent_revisions_match_until_terminal_communication_decision(self):
        script = r'''
import asyncio, os, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
sys.path.insert(0, sys.argv[1])
import numpy as np
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage
from masrubric.agents.supervisor_reasoning_pick_metric import Supervisor
from masrubric.agents.math_solver_gsm8k import MathSolverGsm8k
from masrubric.agents.code_writing_mbpp import CodeWritingMbpp

criterion = {"name": "sum", "detailed_definition": "Check addition", "evaluator_prompt": {"trigger_condition": "addition", "risk_alert": "check sum"}}
flaw = {"metric": "sum", "verdict": "Flawed", "reasoning": "fixture", "suggestion": "check sum"}

async def run(cls, domain, policy, budget):
    os.environ['MASRUBRIC_WITHHOLDING_POLICY'] = policy
    os.environ['MASRUBRIC_CODE_RETRY_BUDGET'] = str(budget)
    s = Supervisor(domain=domain, model='fixture', api_key='EMPTY', base_url='http://127.0.0.1:1/v1',
                   embedding_api_key='EMPTY', embedding_api_base='http://127.0.0.1:1/v1',
                   preloaded_metrics=[criterion], preloaded_embeddings=np.array([[1., 0.]]))
    s._match_metrics = AsyncMock(return_value=[criterion])
    s._calc_score = AsyncMock(return_value=[flaw])
    args = dict(domain=domain, name='Participant_1', role=('Math Solver' if domain == 'gsm8k' else 'Programming Expert'), model='fixture', api_key='EMPTY',
                base_url='http://127.0.0.1:1/v1', supervisor=s, message_history=[])
    if domain == 'gsm8k': args['reflection_time'] = budget
    agent = cls(**args)
    calls = []
    async def completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(model_dump=lambda: {'role': 'assistant', 'content': '1+1=3'})
        return SimpleNamespace(model='fixture', choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    agent._model_client.chat.completions.create = completion
    response = await agent.on_messages([TextMessage(content='What is 1+1?', source='user')], CancellationToken())
    score = s.get_scores({'Participant_1': 'fixture_solver'})[response.chat_message.id]
    assert len(calls) == budget+1
    assert len(score['audit_attempts']) == budget+1
    assert score['is_pruned'] is (policy == 'withhold')
    assert all(a['selected_metrics'] == ['sum'] for a in score['audit_attempts'])
    assert score['audit_attempts'][0]['input_messages'][0]['content'] == 'What is 1+1?'
    if budget:
        assert calls[1]['messages'][-2]['content'] == '1+1=3'
        assert 'external auditor' in calls[1]['messages'][-1]['content']
    await agent._model_client.close()
    await s._model_client.close()
    s.embedding_client.close()
    return calls

async def main():
    for cls, domain in [(MathSolverGsm8k, 'gsm8k'), (CodeWritingMbpp, 'mbpp')]:
        for budget in [0, 1, 3]:
            full = await run(cls, domain, 'withhold', budget)
            revised = await run(cls, domain, 'broadcast', budget)
            assert full == revised, 'Communication switch changed generation/revision prompts or budget'
    print('participant controls passed')
asyncio.run(main())
'''
        env = {k: v for k, v in os.environ.items() if not k.startswith("MASRUBRIC_")}
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT / "paper/runtime")],
                                cwd=ROOT, env=env, text=True, capture_output=True, encoding="utf-8", timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_terminal_policy_and_empty_audit_provenance(self):
        # Isolation prevents importing the other public package named masrubric.
        script = r'''
import asyncio, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
sys.path.insert(0, sys.argv[1])
import numpy as np
from autogen_agentchat.messages import TextMessage
from masrubric.agents.supervisor_reasoning_pick_metric import Supervisor

metric = {"name": "addition", "detailed_definition": "Check addition", "evaluator_prompt": {"trigger_condition": "addition", "risk_alert": "Check sum"}}
bad = {"metric": "addition", "verdict": "Flawed", "reasoning": "wrong sum", "suggestion": "check sum"}

async def make(policy):
    os.environ['MASRUBRIC_WITHHOLDING_POLICY'] = policy
    s = Supervisor(domain='gsm8k', model='fixture', api_key='EMPTY', base_url='http://127.0.0.1:1/v1',
                   embedding_api_key='EMPTY', embedding_api_base='http://127.0.0.1:1/v1',
                   preloaded_metrics=[metric], preloaded_embeddings=np.array([[1., 0.]]), pass_rate=.6)
    return s

async def main():
    for policy, withheld in [('withhold', True), ('broadcast', False)]:
        s = await make(policy)
        s._match_metrics = AsyncMock(return_value=[metric])
        s._calc_score = AsyncMock(return_value=[bad])
        msg = TextMessage(content='1+1=3', source='Participant_1')
        for attempt in [1, 2]:
            passed, judgments, feedback, selected = await s.judge('What is 1+1?', msg, attempt)
            assert not passed and feedback and selected == [metric]
        final = TextMessage(content=msg.content, source=msg.source)
        s.update_scoreboard_with_results(final, judgments)
        score = s.get_scores({'Participant_1': 'solver'})[final.id]
        assert score['is_pruned'] is withheld
        assert score['audit_failed'] is True
        assert len(score['audit_attempts']) == 2
        assert score['audit_attempts'][0]['action'] == 'revise'
        assert score['audit_attempts'][-1]['action'] == ('withhold' if withheld else 'broadcast_after_failed_audit')
        assert bool(s.prune_info([final])) is (not withheld)
        s._match_metrics = AsyncMock(return_value=[])
        s._calc_score = AsyncMock(return_value=[])
        passed, judgments, _, _ = await s.judge('What is 1+1?', msg, 1)
        assert passed
        final = TextMessage(content='2', source=msg.source)
        s.update_scoreboard_with_results(final, judgments)
        score = s.get_scores({'Participant_1': 'solver'})[final.id]
        assert not score['is_pruned']
        assert score['audit_attempts'][0]['empty_cause'] == 'no_selected_criteria'
        assert score['audit_attempts'][0]['empty_policy'] == 'pass'
        assert score['audit_attempts'][0]['pass_rate_observed'] is None
        # Actual per-criterion failure path, not just a mocked empty return.
        s._match_metrics = AsyncMock(return_value=[metric])
        del s._calc_score
        s._model_client.chat.completions.create = AsyncMock(side_effect=ValueError('fixture parse failure'))
        passed, judgments, _, _ = await s.judge('What is 1+1?', msg, 1)
        assert passed and judgments == []
        final = TextMessage(content='2', source=msg.source)
        s.update_scoreboard_with_results(final, judgments)
        score = s.get_scores({'Participant_1': 'solver'})[final.id]
        audit = score['audit_attempts'][0]
        assert audit['empty_cause'] == 'all_selected_audits_failed'
        assert audit['selected_count'] == 1 and audit['judgment_count'] == 0
        assert audit['audit_failures'][0]['attempts'] == 5
        s.reset()  # Earlier attempts must remain in the append-only trace.
        await s._model_client.close()
        s.embedding_client.close()
    events = [json.loads(line) for line in Path(os.environ['MASRUBRIC_MECHANISM_TRACE']).read_text().splitlines()]
    assert sum(e['event_type'] == 'message_finalized' for e in events) == 6
    assert sum(e['event_type'] == 'scoreboard_reset' for e in events) == 2
    # Actual retrieval distinguishes an explicit empty result, unknown names,
    # and a parse failure that recovers through the candidate Top-Q fallback.
    s = await make('withhold')
    s.embedding_client.embeddings.create = lambda **kw: SimpleNamespace(
        model='fixture', usage=None, data=[SimpleNamespace(embedding=[1., 0.])])
    def reply(content):
        return SimpleNamespace(model='fixture', usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    for selection, expected in [('[]', []), ('["unknown"]', []), ('BROKEN', [metric])]:
        raw = json.dumps({'selected_metrics': json.loads(selection)}) if selection != 'BROKEN' else 'not JSON'
        s._model_client.chat.completions.create = AsyncMock(side_effect=[
            reply('{"problem_scenario":["addition"],"agent_action":["sum"]}'), reply(raw)])
        actual = await s._match_metrics('1+1?', '3')
        assert actual == expected
        assert s._selection_trace['fallback'] is (selection == 'BROKEN')
        if selection == '["unknown"]': assert s._selection_trace['unknown_selected_names'] == ['unknown']
    await s._model_client.close()
    s.embedding_client.close()
    print('mechanism assertions passed')
asyncio.run(main())
'''
        with tempfile.TemporaryDirectory() as directory:
            env = {k: v for k, v in os.environ.items() if not k.startswith("MASRUBRIC_")}
            env.update(PYTHONIOENCODING="utf-8", MASRUBRIC_MECHANISM_TRACE=str(Path(directory) / "mechanism.jsonl"))
            result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT / "paper/runtime")],
                                    env=env, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
