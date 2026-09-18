import os
from typing import AsyncGenerator, Sequence, Dict, List, Any, Tuple, Union
import logging
import asyncio
import httpx
import traceback 
from openai import (
    AsyncOpenAI,
    APITimeoutError,
    APIConnectionError,
    DefaultAsyncHttpxClient,
    InternalServerError,
)

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.usage_tracking import record_chat_completion
from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.agents.supervisor import Supervisor
from AgentDropout.agents.agent_mode_helpers import agent_mode_enabled, emit_agent_mode_response

@AgentRegistry.register('CodeWriting_livecode')
class CodeWritingLivecode(BaseChatAgent):
    def __init__(self,
        domain: str,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        supervisor: Supervisor = None,
        role:str = None,
        message_history: List[BaseChatMessage] = None,
    ):
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        description = self.prompt_set.get_description(self.role)
        super().__init__(name=name, description=description)
        
        self.model = model
        self._message_history: List[BaseChatMessage] = message_history if message_history is not None else []
        self._model_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )
        self._system_message = self.prompt_set.get_constraint(self.role) 
        self.role_map = {}
        self.supervisor: Supervisor = supervisor
        
        # LiveCodeBench 是 IO 题目，通常没有内嵌的断言测试
        self.internal_tests = [] 

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    def extract_example(self, prompt_text: str) -> list:
        """
        LiveCodeBench 的题目通常不包含 '>>> assert' 风格的测试。
        为了避免误提取或生成错误的测试代码，这里直接返回空。
        如果需要自测，需要在 System Prompt 中引导模型自己编写测试脚本。
        """
        return []
        
    async def _process_inputs(self, task: str, input_messages: List[TextMessage]) -> Tuple[str, str]:
        system_prompt = self._system_message
        
        MAX_CHAR_LIMIT = 10000
        spatial_str = ""
        for msg in input_messages:
            if msg.source == 'user':
                continue
            
            sender_name = msg.source
            sender_role = self.role_map.get(sender_name, "Unknown Role")
            output_content = msg.content
            
            if len(output_content) > MAX_CHAR_LIMIT:
                display_content = output_content[:MAX_CHAR_LIMIT] + "\n...[Truncated Code/Text]..."
            else:
                display_content = output_content

            spatial_str += f"Agent {sender_name} ({sender_role}) says: {display_content}\n\n"

        user_prompt = f"The task is:\n\n{task}\n"
        if spatial_str:
            user_prompt += f"Previous context from team:\n\n{spatial_str}\n\n"
            
        return system_prompt, user_prompt
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[Union[BaseAgentEvent, BaseChatMessage, Response], None]:
        
        self._message_history.extend(messages)
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")
        
        # 尝试提取测试用例（LiveCode 通常返回空）
        self.internal_tests = self.extract_example(question)

        # Supervisor 剪枝
        pruned_messages = self.supervisor.prune_info(self._message_history) if self.supervisor else self._message_history
        
        system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=pruned_messages)

        if agent_mode_enabled(self):
            yield await emit_agent_mode_response(self, question, max_tokens=2048)
            return

        base_context = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        current_attempt = 0
        max_attempts = 4 
        
        session_metrics = None
        last_agent_content = None
        last_feedback = None
        
        final_response_dict = {}
        final_judgements = []

        while current_attempt < max_attempts:
            current_conversation = list(base_context)
            if current_attempt > 0 and last_agent_content and last_feedback:
                current_conversation.append({"role": "assistant", "content": last_agent_content})
                current_conversation.append({"role": "user", "content": last_feedback})

            try:
                completion = await self._model_client.chat.completions.create(
                    model=self.model,
                    messages=current_conversation,
                    temperature=0.7,
                    max_tokens=2048, 
                )
                record_chat_completion(completion, model=self.model, stage="reasoning", source=self.name)
                
                response_dict = completion.choices[0].message.model_dump()
                current_content = response_dict.get('content') or ''
                last_agent_content = current_content
                
                print(f"\n[DEBUG] Agent '{self.name}' Generated Code Snippet (Attempt {current_attempt + 1}):")
                print("-" * 30)
                print(current_content[:200] + "..." if len(current_content) > 200 else current_content)
                print("-" * 30)
                
                # Fast Track: 仅在有提取到测试用例时运行（LiveCode 默认跳过）
                if self.internal_tests:
                    # (此处保留逻辑结构，但实际上 LiveCodeBench 不会进入这里)
                    pass

                # Supervisor Audit
                temp_message_for_judge = TextMessage(content=current_content, source=self.name)
                
                if self.supervisor and getattr(self.supervisor, 'prune_flag', True):
                    print(f"🔍 [Supervisor] Auditing...")
                    pass_flag, judgements, feedback, used_metrics = await self.supervisor.judge(
                        task=question, 
                        message=temp_message_for_judge,
                        attempt_num=current_attempt + 1,
                        role=self.role,
                        session_metrics=session_metrics 
                    )
                    
                    if session_metrics is None:
                        session_metrics = used_metrics
                    
                    final_response_dict = response_dict
                    final_judgements = judgements
                    
                    if pass_flag:
                        break 

                    if current_attempt < (max_attempts - 1):
                        last_feedback = feedback
                else:
                    final_response_dict = response_dict
                    break

            except (APITimeoutError, APIConnectionError, httpx.ConnectTimeout, InternalServerError) as e:
                logging.warning(f"Agent '{self.name}' API timeout (Attempt {current_attempt + 1})")
                if current_attempt < max_attempts - 1:
                    current_attempt += 1
                    await asyncio.sleep(5)
                    continue
                else:
                    raise e
            except Exception as e:
                logging.error(f"Error for Agent '{self.name}': {e}")
                traceback.print_exc()
                raise e

            current_attempt += 1

        # ---------------------------------------------------------
        
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        if 'completion' in locals():
             usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)

        response_message = TextMessage(
            content=final_response_dict.get('content') or '', 
            source=self.name, 
            models_usage=usage
        )
        
        if self.supervisor and hasattr(self.supervisor, 'update_scoreboard_with_results'):
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
            return Response(chat_message=TextMessage(content="Error: No response generated.", source=self.name))
        return final_response
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._message_history = []
