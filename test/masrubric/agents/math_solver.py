from masrubric.usage import safe_request_usage, tracked_create
import os
from typing import AsyncGenerator, Sequence, Dict, List, Any, Tuple

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from project_datasets.gsm8k_dataset import gsm_get_predict # GSM8K-specific import
from masrubric.tools.coding.python_executor import execute_code_get_return
from masrubric.agents.agent_registry import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from openai import AsyncOpenAI


@AgentRegistry.register('MathSolver')
class MathSolver(BaseChatAgent):
    def __init__(self,
        domain: str,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        supervisor: Supervisor = None,
        role:str = None,
        message_history: List[BaseChatMessage] = [],
        reflection_time: int = 0, 
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
        self.reflection_time = reflection_time 

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
        
    async def _process_inputs(self, task: str, input_messages: List[TextMessage]) -> Tuple[str, str]:
        system_prompt = self._system_message
        user_prompt = self.prompt_set.get_answer_prompt(question=task, role=self.role)

        if self.role == "Math Solver":
            hints = []
            for msg in input_messages:
                if msg.source == 'user':
                    continue
           
                hint = gsm_get_predict(msg.content)
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
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        
        self._message_history.extend(messages)
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")

        pruned_messages = self.supervisor.prune_info(self._message_history) if self.supervisor else self._message_history
        system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=pruned_messages)
        
        current_attempt = 0
    
        max_attempts = self.reflection_time + 1
        
        conversation: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        final_response_dict = {}
        final_judgements = []
        
        while current_attempt < max_attempts:
            completion = await tracked_create(
                self._model_client.chat.completions.create, stage="reasoning", source=self.name, metadata={"agent_attempt": current_attempt + 1},
                model=self.model,
                messages=conversation,
                temperature=0.7,
          
            )
            
            response_dict = completion.choices[0].message.model_dump()
            response_content = response_dict.get('content', '')

  
            if self.role == "Programming Expert":
                answer = execute_code_get_return(response_content.lstrip("```python\n").rstrip("\n```"))
                response_content += f"\nthe answer is {answer}"
                response_dict['content'] = response_content

            conversation.append(response_dict)
            
            temp_message_for_judge = TextMessage(
                content=response_content,
                source=self.name 
            )
            
    
            print(f"\n---------- [Attempt {current_attempt + 1}] Agent '{self.name}' Output ----------")
            print(temp_message_for_judge.content)
            print("--------------------------------------------------")
            
        
            if self.supervisor:
                pass_flag, judgements, feedback = await self.supervisor.judge(
                    task=question, 
                    message=temp_message_for_judge,
                    attempt_num=current_attempt + 1
                )
                
                final_response_dict = response_dict
                final_judgements = judgements
                
                if pass_flag:
                    break

           
                if current_attempt < self.reflection_time:
                    conversation.append({"role": "user", "content": feedback})
            else:
                
                final_response_dict = response_dict
                break

            current_attempt += 1
            
        usage = safe_request_usage(completion)
        response_message = TextMessage(
            content=final_response_dict.get('content', ''), 
            source=self.name, 
            models_usage=usage
        )
        
        if self.supervisor:
            self.supervisor.update_scoreboard_with_results(
                message=response_message, 
                judgements=final_judgements
            )

        self._message_history.append(response_message)
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