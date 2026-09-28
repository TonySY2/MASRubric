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
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.usage_tracking import record_chat_completion
from masrubric.tools.coding.python_executor import MBPPExecutor
from masrubric.agents.agent_registry import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from masrubric.agents.agent_mode_helpers import agent_mode_enabled, emit_agent_mode_response


@AgentRegistry.register('CodeWriting_humaneval')
class CodeWritingHumaneval(BaseChatAgent):
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
                function_call = line[4:].strip()
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")
        return results
        
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

            # 这里移除了所有的内部测试逻辑，只做单纯的 Prompt 拼接
            # 真正的测试移到了 on_messages_stream 的生成循环里
            spatial_str += f"Agent {sender_name} ({sender_role}) says: {display_content}\n\n"

        user_prompt = f"The task is:\n\n{task}\n"
        if spatial_str:
            user_prompt += f"Previous context:\n\n{spatial_str}\n\n"
            
        return system_prompt, user_prompt
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[Union[BaseAgentEvent, BaseChatMessage, Response], None]:
        
        self._message_history.extend(messages)
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")
        
        self.internal_tests = self.extract_example(question)
        if self.internal_tests:
            print(f"[{self.name}] Detected {len(self.internal_tests)} internal test cases.")

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
                
                # --- [新增调试打印] 看看它到底生成了啥 ---
                print(f"\n[DEBUG] Agent '{self.name}' Generated Code Snippet (Attempt {current_attempt + 1}):")
                print("-" * 30)
                print(current_content[:200] + "..." if len(current_content) > 200 else current_content)
                print("-" * 30)
                
                passed_internal_tests = False
                
                # Fast Track Logic
                if self.internal_tests and self.role not in ['Normal Programmer', 'Stupid Programmer']:
                    try:
                        if "```python" in current_content:
                            code_to_test = current_content.split("```python")[1].split("```")[0].strip()
                        else:
                            code_to_test = current_content.strip()
                        
                        # 把前面定义的函数签名（Prompt里给的）拼上去，防止有些模型只写了 body
                        # 这里 HumanEval 比较特殊，Prompt 里自带了签名 def strlen(...)
                        # 如果模型复读了 def strlen，那是最好的。
                        
                        test_script = "\n".join(self.internal_tests)
                        is_solved, feedback, _ = MBPPExecutor().execute(code_to_test, [test_script], timeout=5)
                        
                        if is_solved:
                            print(f"🚀 [Fast Track] Passed internal tests! (Code valid)")
                            passed_internal_tests = True
                        else:
                            print(f"⚠️ [Self-Check Failed] Internal tests failed.")
                    except Exception as e:
                        print(f"[Fast Track Error] {e}")

                if passed_internal_tests:
                    final_response_dict = response_dict
                    final_response_dict['content'] += "\n\n[Status: Internal Tests Passed]"
                    break 

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
