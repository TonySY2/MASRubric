#masrubric/agents/math_solver_aqua.py

import os
from typing import AsyncGenerator, Sequence, Dict, List, Any, Tuple, Union
import json
import datetime

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from project_datasets.aqua_dataset import aqua_get_predict
from masrubric.agents.agent_registry import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from openai import AsyncOpenAI
#from transformers import AutoTokenizer

from openai import APITimeoutError, APIConnectionError
import logging
import asyncio
import httpx

@AgentRegistry.register('MathSolver_aqua')
class MathSolverAqua(BaseChatAgent):
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
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    

    def _capture_trajectory(self, task: str, output: str) -> Dict[str, Any]:

        return {
            "task": task,
            "agent_name": self.name,
            "agent_role": self.role,
            "agent_role_description": self.description, 
            "output": output,
            "dataset_name": "aqua",
            "source_model": self.model,
        }
        
    async def _process_inputs(self, task: str, input_messages: List[TextMessage]) -> Tuple[str, str]:
        system_prompt = self._system_message
        user_prompt = self.prompt_set.get_answer_prompt(question=task, role=self.role)

        if self.role == "Math Solver":
            hints = []
            for msg in input_messages:
                if msg.source == 'user':
                    continue
                hint = aqua_get_predict(msg.content)
                if hint:
                    hints.append(hint)
            if hints:
                user_prompt += f"(Hint: The answer from other agents are list: {' '.join(hints)})."
        else:
            spatial_str = ""
            for msg in input_messages:
                if msg.source == 'user':
                    continue
                sender_role = self.role_map.get(msg.source, "an assistant")
                spatial_str += f"Agent {msg.source} as a {sender_role} his answer to this question is:\n\n{msg.content}\n\n"
            if spatial_str:
                user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str}\n\n"
                
        return system_prompt, user_prompt
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[Union[BaseAgentEvent, BaseChatMessage, Response], None]:
        
        self._message_history.extend(messages)
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")
        if not question:
            raise ValueError("Input messages must contain a user message for the task.")

        pruned_messages = self.supervisor.prune_info(self._message_history) if self.supervisor else self._message_history
        system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=pruned_messages)
        
        
        completion_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            # "extra_body": {
            #     "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            #     "chat_template_kwargs": {"enable_thinking": False}
            # }
        }


        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                completion = await self._model_client.chat.completions.create(**completion_params)
                
                response = completion.choices[0].message
                usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)
                response_message = TextMessage(content=response.content, source=self.name, models_usage=usage)
                
                self._message_history.append(response_message)
                
                # math_solver_aqua.py

                
     
                if self.supervisor:
                    if self.supervisor.supervisor_mode == 'collect':
            
                        trajectory = self._capture_trajectory(task=question, output=response_message.content)
                        
          
                        full_task_json = getattr(self.supervisor, 'current_full_task_json', "")
                        
             
                        await self.supervisor.collect_trajectory(trajectory, response_message, full_task_json)
                        
                    elif self.supervisor.supervisor_mode == 'prune':
                        await self.supervisor.update_scoreboard(task=question, message=response_message)
                # ==========================================================
                
                yield Response(chat_message=response_message)
                
                return

            except (APITimeoutError, APIConnectionError, httpx.ConnectTimeout) as e:
                logging.warning(f"Agent '{self.name}' API request failed (Attempt {attempt + 1}/{max_retries}). Error: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(f"Agent '{self.name}' failed after {max_retries} attempts. Raising the exception.")
                    raise e
            except Exception as e:
                logging.error(f"An unexpected error occurred for Agent '{self.name}': {e}")
                if hasattr(e, 'body') and e.body:
                    print("--- Server Response Body ---")
                    try:
                        print(json.dumps(e.body, indent=2))
                    except:
                        print(e.body)
                raise e
    
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