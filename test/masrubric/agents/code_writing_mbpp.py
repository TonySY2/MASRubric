from masrubric.usage import safe_request_usage, tracked_create
import os
from typing import AsyncGenerator, Sequence, Dict, List, Any, Tuple
import traceback 

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.tools.coding.python_executor import MBPPExecutor
from masrubric.agents.agent_registry import AgentRegistry
from masrubric.agents.supervisor import Supervisor
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError
import httpx
import asyncio
from masrubric.prompt.mbpp_prompt_set import FEW_SHOT_DATA_MBPP

@AgentRegistry.register('CodeWriting_mbpp')
class CodeWritingMbpp(BaseChatAgent):
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
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._system_message = self.prompt_set.get_constraint(self.role) 
        self.role_map = {}
        self.reflection_time = reflection_time
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

            if output_content.startswith("```python") and self.role not in ['Normal Programmer', 'Stupid Programmer']:
                try:
                    code_to_test = output_content.split("```python")[1].split("```")[0].strip()
                except:
                    code_to_test = output_content.strip()
                
  
                if self.internal_tests:
                    test_str = "\n".join(self.internal_tests)
                    is_solved, feedback, _ = MBPPExecutor().execute(code_to_test, [test_str], timeout=5)
                    
                    if is_solved:
                        return "is_solved", output_content

                    spatial_str += f"Agent {sender_name} as a {sender_role}:\n\nThe code written by the agent is:\n\n{display_content}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                else:
                    spatial_str += f"Agent {sender_name} as a {sender_role}:\n\nThe code written by the agent is:\n\n{display_content}\n\n(No internal tests available)\n\n"
            else:
                spatial_str += f"Agent {sender_name} as a {sender_role} provides the following info: {display_content}\n\n"

        user_prompt = f"The task is:\n\n{task}\n"
        if spatial_str:
            user_prompt += f"At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
            
        return system_prompt, user_prompt
    
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        
        self._message_history.extend(messages)
        
        question = next((msg.content for msg in self._message_history if msg.source == 'user'), "")
        
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

 
        base_context = [{"role": "system", "content": system_prompt}]

        if self.role in FEW_SHOT_DATA_MBPP:
            for example_q, example_a in FEW_SHOT_DATA_MBPP[self.role]:
                base_context.append({"role": "user", "content": example_q})
                base_context.append({"role": "assistant", "content": example_a})

        base_context.append({"role": "user", "content": user_prompt})


        
        current_attempt = 0
        max_attempts = self.reflection_time + 1
        

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
                completion = await tracked_create(
                    self._model_client.chat.completions.create, stage="reasoning", source=self.name, metadata={"agent_attempt": current_attempt + 1},
                    model=self.model,
                    messages=current_conversation,
                    temperature=0.7,
                    max_tokens=2048, 
                )
                response_dict = completion.choices[0].message.model_dump()
                current_content = response_dict.get('content', '')
  
                last_agent_content = current_content
                
                temp_message_for_judge = TextMessage(
                    content=current_content,
                    source=self.name 
                )
                
       
                if self.supervisor and getattr(self.supervisor, 'prune_flag', True):
                    print(f"\n---------- [Attempt {current_attempt + 1}] Agent '{self.name}' Output ----------")
                    
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
            
            except (APITimeoutError, APIConnectionError, httpx.ConnectTimeout) as e:
                if current_attempt < max_attempts - 1:
                    # Network failures consume the same bounded attempt budget.
                    current_attempt += 1
                    await asyncio.sleep(5)
                    continue
                else:
                    raise e
            except Exception as e:
                print(f"!!! CRITICAL EXCEPTION IN AGENT {self.name} !!!")
                traceback.print_exc()
                raise e

            current_attempt += 1

        # ---------------------------------------------------------
            
        usage = None
        if 'completion' in locals():
             usage = safe_request_usage(completion)

        response_message = TextMessage(
            content=final_response_dict.get('content', ''), 
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
            raise AssertionError("The stream should have returned the final result.")
        return final_response
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._message_history = []
