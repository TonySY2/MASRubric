"""Offline checks of the fixed graph's scheduling and visibility contract."""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test"))

from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from masrubric.teams import FixedDAGTeam, create_team


class Gate:
    def __init__(self, rejected=(), enabled=True):
        self.rejected = set(rejected)
        self.prune_flag = enabled

    def prune_info(self, messages):
        return [message for message in messages if message.content not in self.rejected]


class RecordingAgent:
    def __init__(self, name, gate=None):
        self.name = name
        self.supervisor = gate
        self.inputs = []
        self.history = []
        self.calls = 0

    async def on_reset(self, cancellation_token):
        self.history = []

    async def on_messages(self, messages, cancellation_token):
        self.history.extend(messages)
        self.inputs.append([message.content for message in self.history])
        self.calls += 1
        return Response(chat_message=TextMessage(content=f"{self.name}:{self.calls}", source=self.name))


class FixedTeamTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_forward_edges_and_no_selector(self):
        agents = [RecordingAgent(name) for name in "ABCDE"]
        team = create_team(framework="fixed", participants=agents, model_client=object(),
                           termination_condition=object(), selector_prompt="unused")
        self.assertEqual(len(team.spatial_edges), 10)
        await team.run(task="problem")
        for index, agent in enumerate(agents):
            self.assertEqual(agent.inputs, [["problem"] + [f"{name}:1" for name in "ABCDE"[:index]]])
        self.assertEqual([message.content for message in team.final_messages], [f"{name}:1" for name in "ABCDE"])

    async def test_rejection_hidden_from_descendants_and_final(self):
        gate = Gate({"A:1"})
        agents = [RecordingAgent(name, gate) for name in "ABC"]
        team = FixedDAGTeam(agents)
        await team.run(task="problem")
        self.assertEqual(agents[1].inputs[0], ["problem"])
        self.assertEqual(agents[2].inputs[0], ["problem", "B:1"])
        self.assertEqual([m.content for m in team.final_messages], ["B:1", "C:1"])

    async def test_baseline_retains_all_outputs(self):
        gate = Gate({"A:1"}, enabled=False)
        agents = [RecordingAgent(name, gate) for name in "AB"]
        team = FixedDAGTeam(agents)
        await team.run(task="problem")
        self.assertEqual(agents[1].inputs[0], ["problem", "A:1"])
        self.assertEqual(len(team.final_messages), 2)

    async def test_two_rounds_use_only_current_and_previous_forward_predecessors(self):
        agents = [RecordingAgent(name) for name in "ABC"]
        team = FixedDAGTeam(agents, fixed_rounds=2)
        await team.run(task="problem")
        self.assertEqual(agents[0].inputs[1], ["problem"])
        self.assertEqual(agents[1].inputs[1], ["problem", "A:2", "A:1"])
        self.assertEqual(agents[2].inputs[1], ["problem", "A:2", "B:2", "A:1", "B:1"])
        self.assertEqual([m.content for m in team.final_messages], ["A:2", "B:2", "C:2"])

    async def test_reset_clears_final_messages(self):
        agent = RecordingAgent("A")
        team = FixedDAGTeam([agent])
        await team.run(task="first")
        await team.reset()
        self.assertEqual(team.final_messages, [])
        await team.run(task="second")
        self.assertEqual(agent.inputs[-1], ["second"])

    async def test_cancelled_task_does_not_call_agents(self):
        agent = RecordingAgent("A")
        team = FixedDAGTeam([agent])
        token = CancellationToken()
        token.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await team.run(task="problem", cancellation_token=token)
        self.assertEqual(agent.calls, 0)
        await team.reset()

    def test_dynamic_factory_forwards_autogen_options_unchanged(self):
        agents = [RecordingAgent("A")]
        client = object()
        termination = object()
        with patch("masrubric.teams.SelectorGroupChat") as constructor:
            result = create_team(participants=agents, framework="dynamic", model_client=client,
                                 termination_condition=termination, selector_prompt="select",
                                 allow_repeated_speaker=True)
        constructor.assert_called_once_with(participants=agents, model_client=client,
                                            termination_condition=termination, selector_prompt="select",
                                            allow_repeated_speaker=True)
        self.assertIs(result, constructor.return_value)

    def test_invalid_graph_arguments(self):
        for rounds in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                FixedDAGTeam([RecordingAgent("A")], fixed_rounds=rounds)
        with self.assertRaises(ValueError):
            FixedDAGTeam([])
        with self.assertRaises(ValueError):
            FixedDAGTeam([RecordingAgent("A"), RecordingAgent("A")])


if __name__ == "__main__":
    unittest.main()
