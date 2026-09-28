import os
from typing import AsyncGenerator, Sequence, Dict, List, Any, Tuple

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.usage_tracking import record_chat_completion
from masrubric.tools.coding.python_executor import PyExecutor
from masrubric.agents.agent_registry import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from masrubric.agents.agent_mode_helpers import agent_mode_enabled, emit_agent_mode_response
from openai import AsyncOpenAI

@AgentRegistry.register('CodeWriting')
class CodeWriting(BaseChatAgent):
    def __init__(self,
        domain: str,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        supervisor: Supervisor = None,
        role:str = None,
        message_history: List[BaseChatMessage] = [],
    ):
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        description = self.prompt_set.get_description(self.role)
        super().__init__(name=name, description=description)
        
        self.model = model
        self._message_history: List[BaseChatMessage] = message_history
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._system_message = self.prompt_set.get_constraint(self.role) 
        self.role_map = {}
        self.supervisor: Supervisor = supervisor
        self.internal_tests = []

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    def extract_example(self, prompt_text: str) -> list:
        lines = (line.strip() for line in prompt_text.split('\n') if line.strip())
        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")
        return results
        
    async def _process_inputs(self, task: str, input_messages: List[TextMessage]) -> Tuple[str, str]:
        system_prompt = self._system_message
        
        spatial_str = ""
        for msg in input_messages:
            if msg.source == 'user':
                continue
            
            sender_name = msg.source
            sender_role = self.role_map.get(sender_name, "Unknown Role")
            output_content = msg.content

            if output_content.startswith("```python") and self.role not in ['Normal Programmer', 'Stupid Programmer']:
                code_to_test = output_content.lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, _ = PyExecutor().execute(code_to_test, self.internal_tests, timeout=10)
                
                if is_solved and len(self.internal_tests):
                    return "is_solved", output_content

                spatial_str += f"Agent {sender_name} as a {sender_role}:\n\nThe code written by the agent is:\n\n{output_content}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {sender_name} as a {sender_role} provides the following info: {output_content}\n\n"

        user_prompt = f"The task is:\n\n{task}\n"
        if spatial_str:
            user_prompt += f"At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
            
        return system_prompt, user_prompt
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        
        self._message_history.extend(messages)
        
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")
        if not question:
            raise ValueError("The input messages should contain a user message as the task.")

        self.internal_tests = self.extract_example(question)

        pruned_messages = self.supervisor.prune_info(self._message_history) if self.supervisor else self._message_history
        
        system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=pruned_messages)

        if system_prompt == "is_solved":
            response_message = TextMessage(content=user_prompt, source=self.name)
            self._message_history.append(response_message)
            if self.supervisor:
                await self.supervisor.update_scoreboard(task=user_prompt, message=response_message)
            yield Response(chat_message=response_message)
            return

        if agent_mode_enabled(self):
            yield await emit_agent_mode_response(self, question, max_tokens=2048)
            return

        completion = await self._model_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        record_chat_completion(completion, model=self.model, stage="reasoning", source=self.name)
        response = completion.choices.message
        usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)
        response_message = TextMessage(content=response.content, source=self.name, models_usage=usage)
        
        self._message_history.append(response_message)
        if self.supervisor:
            await self.supervisor.update_scoreboard(task=user_prompt, message=response_message)
        
        yield Response(chat_message=response_message)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message
        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")
        return final_response
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._message_history = []
