import os
from typing import AsyncGenerator, Sequence, Dict, List, Any, Tuple
import re

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.agents.agent_registry import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from masrubric.tools.search.wiki import search_wiki_main
from openai import AsyncOpenAI

def find_strings_between_pluses(text):
    return re.findall(r'\@(.*?)\@', text)

@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(BaseChatAgent):
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
        self._system_message = self.prompt_set.get_analyze_constraint(self.role)
        self.role_map = {}
        self.supervisor: Supervisor = supervisor
        self.wiki_summary = ""

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
        
    async def _process_inputs(self, task: str, input_messages: List[TextMessage]) -> Tuple[str, str]:
        system_prompt = self._system_message
        
        user_prompt = f"The task is: {task}\n"
        spatial_str = ""
        for msg in input_messages:
            if msg.source == 'user':
                continue
                
            sender_name = msg.source
            sender_role = self.role_map.get(sender_name, "Unknown Role")
            output_content = msg.content
            
            if self.role == 'Wiki Searcher' and sender_role == 'Knowlegable Expert':
                queries = find_strings_between_pluses(output_content)
                wiki = await search_wiki_main(queries)
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    user_prompt += f"The key entities of the problem are explained in Wikipedia as follows:{self.wiki_summary}"

            if 'None.' in output_content:
                continue
            spatial_str += f"Agent {sender_name}, role is {sender_role}, output is:\n\n {output_content}\n\n"

        if spatial_str:
            user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str}\n\n"
            
        return system_prompt, user_prompt
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        
        self._message_history.extend(messages)
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")
        if not question:
            raise ValueError("Input messages must contain a user message for the task.")

        pruned_messages = self.supervisor.prune_info(self._message_history) if self.supervisor else self._message_history
        system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=pruned_messages)

        completion = await self._model_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
        )
        response = completion.choices.message
        usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)

        if self.wiki_summary:
            response.content += f"\n\n{self.wiki_summary}"
            self.wiki_summary = ""

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