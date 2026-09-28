import os
from typing import AsyncGenerator, Sequence, Dict, List, Any




from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, RequestUsage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from masrubric.prompt.prompt_set_registry import PromptSetRegistry






from project_datasets.gsm8k_dataset import gsm_get_predict
from masrubric.tools.coding.python_executor import execute_code_get_return
from masrubric.agents.agent_registry import AgentRegistry
from autogen_core.models import UserMessage, SystemMessage, RequestUsage
from masrubric.agents.supervisor import Supervisor
from openai import AsyncOpenAI
#from transformers import AutoTokenizer


@AgentRegistry.register('MathSolver')
class MathSolver(BaseChatAgent):
    def __init__(self,
        domain: str,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        # id: str | None =None,
        supervisor: Supervisor = None,
        role:str = None,
        message_history: List[BaseChatMessage] = [],
    ):
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        description = self.prompt_set.get_description(self.role)
        super().__init__(name=name, description=description)
        # self._model_context = UnboundedChatCompletionContext()
        self.model = model
        self._message_history: List[BaseChatMessage] = message_history
        # self._model_client = OpenAIChatCompletionClient(
        #     model=model,
        #     api_key=api_key,
        #     base_url=base_url,
        #     temperature=0.7,
        # )
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._system_message = self.prompt_set.get_constraint(self.role) 
        self.role_map = {}
        self.supervisor: Supervisor = supervisor
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
        
    # def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
    async def _process_inputs(self, task: str, input_messages: List[TextMessage])->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self._system_message
        
        user_prompt = self.prompt_set.get_answer_prompt(question=task, role=self.role) 
        if self.role == "Math Solver":
            user_prompt += "(Hint: The answer is near to"
            for msg in input_messages:
                if msg.source == 'user':
                    continue
                user_prompt += " " + gsm_get_predict(msg.content)
            user_prompt += ")."
        else:
            spatial_str = ""
            for msg in input_messages:
                if msg.source == 'user':
                    continue
                # spatial_str += f"Agent {msg.source} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
                role_desc = f"a {self.role_map[msg.source]}" if msg.source in self.role_map else "an assistant"
                spatial_str += f"Agent {msg.source} as {role_desc} his answer to this question is:\n\n{msg.content}\n\n" 
            user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
            # user_prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        
        # for msg in messages:
            # await self._model_context.add_message(msg.to_model_message())
        self._message_history.extend(messages)
        
        # all_massages = await self._model_context.get_messages()
        
        # for msg in self._message_history:
        #     print(type(msg), msg.source)
        
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")
        if question == "":
            raise ValueError("The input messages should contain a user message as the question.")

        # Get conversation history
        if self.supervisor is not None:
            pruned_messages = self.supervisor.prune_info(self._message_history)
            system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=pruned_messages)
        else:
            system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=self._message_history)
        

        completion = await self._model_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            # extra_body={
            #     "stop_token_ids":[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            #     "chat_template_kwargs": {"enable_thinking": False}}
        )
        response = completion.choices[0].message

        # Create usage metadata
        usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)

        response_content = response.content
        if self.role == "Programming Expert":
            answer = execute_code_get_return(response_content.lstrip("```python\n").rstrip("\n```"))
            response_content += f"\nthe answer is {answer}"
        response.content = response_content
        # Add response to model context

        response_message = TextMessage(content=response.content, source=self.name, models_usage=usage)
        # await self._model_context.add_message(response_message)
        self._message_history.append(response_message)
        
        if self.supervisor is not None:
            await self.supervisor.update_scoreboard(task=user_prompt, message=response_message)
        
        # Yield the final response
        yield Response(
            chat_message=response_message,
        )
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message

        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")

        return final_response
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        # await self._model_context.clear()
        self._message_history = []
    