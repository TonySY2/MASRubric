#--- START OF FILE math_solver_aqua.py ---

import os
from typing import AsyncGenerator, Sequence, Dict, List, Any, Tuple, Union
import json
import logging
import asyncio
import httpx

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from project_datasets.aqua_dataset import aqua_get_predict 
from AgentDropout.agents.agent_registry import AgentRegistry
from AgentDropout.usage_tracking import record_chat_completion
from AgentDropout.agents.agent_mode_helpers import agent_mode_enabled, emit_agent_mode_response
# 注意这里引用的是基类类型，实际运行时会传入 supervisor_reasoning_pick_metric 实例
from AgentDropout.agents.supervisor import Supervisor 
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    DefaultAsyncHttpxClient,
)
import traceback 

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
        message_history: List[BaseChatMessage] = None,
        reflection_time: int = 3,
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
        self.reflection_time = reflection_time
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
        
    async def _process_inputs(self, task: str, input_messages: List[TextMessage]) -> Tuple[str, str]:
        system_prompt = self._system_message
        user_prompt = self.prompt_set.get_answer_prompt(question=task, role=self.role)

        # AQuA 特有的逻辑：如果是 Math Solver，尝试提取其他 Agent 的答案作为 Hint
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
             # 容错处理，防止空问题
             pass 

        if agent_mode_enabled(self):
            yield await emit_agent_mode_response(self, question, max_tokens=2048)
            return

        # 使用 Supervisor 过滤历史消息 (Pruning)
        pruned_messages = self.supervisor.prune_info(self._message_history) if self.supervisor else self._message_history
        system_prompt, user_prompt = await self._process_inputs(task=question, input_messages=pruned_messages)
        
        current_attempt = 0
        max_attempts = self.reflection_time + 1
        
        # [新增] 用于保存本轮对话中锁定的指标
        session_metrics = None 
        
        base_context: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
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
                    max_tokens=2048,  # <--- 限制它最多只能生成 2048 个 token，防止输出死循环
                )
                
                record_chat_completion(completion, model=self.model, stage="reasoning", source=self.name)
                response_dict = completion.choices[0].message.model_dump()
                
                # 获取当前生成内容
                current_content = response_dict.get('content', '')
                last_agent_content = current_content

                # ======================================================================
                # [核心修改] 打印每一轮的生成内容 (Experiment Analysis 3 Requirement)
                # ======================================================================
                print(f"\n{'-'*30}")
                print(f"📝 [Attempt {current_attempt + 1}] Agent '{self.name}' Generated Content:")
                print(f"{'-'*30}")
                print(current_content)
                print(f"{'-'*50}\n")
                # ======================================================================

                temp_message_for_judge = TextMessage(
                    content=current_content,
                    source=self.name 
                )
                
                # ---------------------------------------------------------
                # [核心逻辑]：带 RAG 审计与重试的生成循环
                # ---------------------------------------------------------
                # 检查 prune_flag，如果为 False (保底模式)，则跳过审计
                if self.supervisor and getattr(self.supervisor, 'prune_flag', True):
                    print(f"\n---------- [Attempt {current_attempt + 1}] Agent '{self.name}' Auditing ----------")
                    
                    # [修改] 接收第4个返回值，并传入 session_metrics
                    pass_flag, judgements, feedback, used_metrics = await self.supervisor.judge(
                        task=question, 
                        message=temp_message_for_judge,
                        attempt_num=current_attempt + 1,
                        role=self.role,
                        session_metrics=session_metrics # 传入当前保存的指标
                    )
                    
                    # [新增] 如果是第一轮（或者 session_metrics 为空），保存 metrics 供后续复用
                    if session_metrics is None:
                        session_metrics = used_metrics
                    
                    final_response_dict = response_dict
                    final_judgements = judgements
                    
                    if pass_flag:
                        print(f"Agent '{self.name}' passed audit.")
                        break 

                    # 未通过：将 Supervisor 的反馈作为 User 消息加入上下文
                    if current_attempt < self.reflection_time:
                        print(f"Agent '{self.name}' failed audit. Retrying with feedback...")
                        last_feedback = feedback
                else:
                    # 无 Supervisor 或 保底模式，直接接受
                    final_response_dict = response_dict
                    break
            
            except (APITimeoutError, APIConnectionError, httpx.ConnectTimeout, APIStatusError) as e:
                status_code = getattr(e, "status_code", None)
                is_retryable_status = status_code in {429, 500, 502, 503, 504}
                if isinstance(e, (APITimeoutError, APIConnectionError, httpx.ConnectTimeout)) or is_retryable_status:
                    logging.warning(
                        f"Agent '{self.name}' API request failed on attempt {current_attempt + 1}/{max_attempts}. "
                        f"status={status_code} error={e}"
                    )
                    if current_attempt < max_attempts - 1:
                        current_attempt += 1
                        await asyncio.sleep(5)
                        continue
                raise e
            except Exception as e:
                logging.error(f"Error for Agent '{self.name}': {e}")
                print(f"!!! CRITICAL EXCEPTION IN AGENT {self.name} !!!")
                traceback.print_exc() 
                raise e

            current_attempt += 1
            
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        if 'completion' in locals():
            usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)
        response_message = TextMessage(
            content=final_response_dict.get('content', ''), 
            source=self.name, 
            models_usage=usage
        )
        
        # 结果回填
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
            raise AssertionError("The stream should have returned the final result.")
        return final_response
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._message_history = []
