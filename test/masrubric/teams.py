"""Dynamic AutoGen teams and the historical FullConnected fixed DAG.

The fixed scheduler follows the original Graph/GuardedGraph implementation:
candidate edges are considered in participant order and cycle-forming edges
are omitted. With the FullConnected mask this gives every edge i -> j for
i < j. The original experiment used five participants and one round. On later
rounds the same forward edges read the previous round's outputs as temporal
context. The final decision is performed by the benchmark runner.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import CancellationToken


class FixedDAGTeam:
    """Execute all forward DAG edges, with audit/retry handled by participants.

    Chat histories are reset before each node execution so only its graph
    predecessors are visible. Rejected messages are removed before subsequent
    nodes run and before final aggregation. Dynamic chat termination conditions
    do not shorten a fixed graph traversal.
    """

    def __init__(self, participants: Sequence[Any], *, fixed_rounds: int = 1) -> None:
        if not participants:
            raise ValueError("A fixed DAG needs at least one participant.")
        if isinstance(fixed_rounds, bool) or not isinstance(fixed_rounds, int) or fixed_rounds < 1:
            raise ValueError("fixed_rounds must be a positive integer.")
        names = [participant.name for participant in participants]
        if len(set(names)) != len(names):
            raise ValueError("Participant names must be unique.")
        self.participants = list(participants)
        self.fixed_rounds = fixed_rounds
        self.spatial_edges = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        self.temporal_edges = list(self.spatial_edges)
        self.final_messages: list[BaseChatMessage] = []
        self._running = False

    @staticmethod
    def _retained(participant: Any, messages: Sequence[BaseChatMessage]) -> list[BaseChatMessage]:
        supervisor = getattr(participant, "supervisor", None)
        if supervisor is not None and getattr(supervisor, "prune_flag", True):
            return list(supervisor.prune_info(list(messages)))
        return list(messages)

    async def reset(self) -> None:
        if self._running:
            raise RuntimeError("Cannot reset a running fixed DAG.")
        self.final_messages = []
        for participant in self.participants:
            await participant.on_reset(CancellationToken())

    async def run_stream(
        self,
        *,
        task: str | BaseChatMessage | Sequence[BaseChatMessage],
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]:
        if self._running:
            raise RuntimeError("The fixed DAG is already running.")
        if isinstance(task, str):
            task_messages = [TextMessage(content=task, source="user")]
        elif isinstance(task, BaseChatMessage):
            task_messages = [task]
        else:
            task_messages = list(task)
        if not task_messages:
            raise ValueError("task must contain at least one message.")

        token = cancellation_token or CancellationToken()
        self._running = True
        self.final_messages = []
        transcript: list[BaseAgentEvent | BaseChatMessage] = list(task_messages)
        previous_round: list[BaseChatMessage] = []
        try:
            for message in task_messages:
                yield message
            for _ in range(self.fixed_rounds):
                current_round: list[BaseChatMessage] = []
                for index, participant in enumerate(self.participants):
                    if token.is_cancelled():
                        raise asyncio.CancelledError()
                    # The original fixed graph exposes current spatial inputs
                    # followed by previous-round temporal inputs, both i < j.
                    predecessors = current_round + previous_round[:index]
                    inputs = task_messages + self._retained(participant, predecessors)
                    await participant.on_reset(token)
                    response = await participant.on_messages(inputs, token)
                    if not isinstance(response, Response):
                        raise TypeError("A fixed DAG participant must return an AutoGen Response.")
                    for event in response.inner_messages or []:
                        transcript.append(event)
                        yield event
                    current_round.append(response.chat_message)
                    transcript.append(response.chat_message)
                    yield response.chat_message
                previous_round = current_round
            self.final_messages = [
                message
                for participant, message in zip(self.participants, previous_round)
                if self._retained(participant, [message])
            ]
            self._running = False
            yield TaskResult(messages=transcript, stop_reason="Fixed DAG traversal completed")
        finally:
            self._running = False

    async def run(self, *, task: str | BaseChatMessage | Sequence[BaseChatMessage],
                  cancellation_token: CancellationToken | None = None) -> TaskResult:
        result = None
        async for event in self.run_stream(task=task, cancellation_token=cancellation_token):
            if isinstance(event, TaskResult):
                result = event
        if result is None:
            raise RuntimeError("The fixed DAG did not produce a TaskResult.")
        return result


def create_team(
    *,
    framework: str = "dynamic",
    participants: Sequence[Any],
    fixed_rounds: int | None = None,
    **dynamic_options: Any,
) -> SelectorGroupChat | FixedDAGTeam:
    """Keep AutoGen behavior unchanged or select the fixed FullConnected DAG."""
    if framework == "dynamic":
        return SelectorGroupChat(participants=participants, **dynamic_options)
    if framework == "fixed":
        return FixedDAGTeam(participants, fixed_rounds=1 if fixed_rounds is None else fixed_rounds)
    raise ValueError(f"Unknown framework: {framework}")
