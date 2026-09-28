from masrubric.usage import safe_request_usage, tracked_create
from masrubric.agents.agent_registry import AgentRegistry
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage, SystemMessage
from typing import List, Dict
from openai import AsyncOpenAI
from masrubric.tools.coding.python_executor import PyExecutor
#from transformers import AutoTokenizer
from masrubric.prompt.mbpp_prompt_set import FEW_SHOT_DATA_MBPP



@AgentRegistry.register('FinalWriteCodeMBPP')
class FinalWriteCodeMBPP:
    def __init__(self, name: str, model: str, api_key: str, base_url: str, domain: str):
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.name = name
        self.domain = domain
        self.prompt_set = PromptSetRegistry.get(self.domain)
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()

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

    def _process_inputs(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str):
        system_prompt = f"{self.role}.\n{self.constraint}"
        spatial_str = ""
        internal_tests = self.extract_example(task)

        for msg in history_messages:
            if msg.source == 'user':
                continue

            sender_name = msg.source
            sender_role = role_map.get(sender_name, "Unknown Role")
            output_content = msg.content

            if output_content.startswith("```python") and output_content.endswith("```"):
                code_to_test = output_content.lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, _ = PyExecutor().execute(code_to_test, internal_tests, timeout=10)
                spatial_str += f"Agent {sender_name} as a {sender_role}:\n\nThe code written by the agent is:\n\n{output_content}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {sender_name} as a {sender_role} provides the following info: {output_content}\n\n"
        
        user_prompt = f"The task is:\n\n{task}\nAt the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
        return system_prompt, user_prompt
    

    
    async def run_decision(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str) -> TextMessage:
        system_prompt, user_prompt = self._process_inputs(history_messages, role_map, task)
        completion = await tracked_create(
            self._model_client.chat.completions.create, stage="final_decision", source=self.name,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message
        usage = safe_request_usage(completion)
        response_message = TextMessage(content=response.content, source=self.name, models_usage=usage)
        return response_message

    
    
@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode:
    def __init__(self, name: str, model: str, api_key: str, base_url: str, domain: str):
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.name = name
        self.domain = domain
        self.prompt_set = PromptSetRegistry.get(self.domain)
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()

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

    def _process_inputs(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str):
        system_prompt = f"{self.role}.\n{self.constraint}"
        spatial_str = ""
        internal_tests = self.extract_example(task)

        for msg in history_messages:
            if msg.source == 'user':
                continue

            sender_name = msg.source
            sender_role = role_map.get(sender_name, "Unknown Role")
            output_content = msg.content

            if output_content.startswith("```python") and output_content.endswith("```"):
                code_to_test = output_content.lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, _ = PyExecutor().execute(code_to_test, internal_tests, timeout=10)
                spatial_str += f"Agent {sender_name} as a {sender_role}:\n\nThe code written by the agent is:\n\n{output_content}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {sender_name} as a {sender_role} provides the following info: {output_content}\n\n"
        
        user_prompt = f"The task is:\n\n{task}\nAt the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
        return system_prompt, user_prompt
    
    async def run_decision(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str) -> TextMessage:
        system_prompt, user_prompt = self._process_inputs(history_messages, role_map, task)
        completion = await tracked_create(
            self._model_client.chat.completions.create, stage="final_decision", source=self.name,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, 
        )
        response = completion.choices[0].message
        usage = safe_request_usage(completion)
        response_message = TextMessage(content=response.content, source=self.name, models_usage=usage)
        return response_message


@AgentRegistry.register('FinalRefer')
class FinalRefer():
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        domain: str,
    ):


        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.name = name
        self.domain = domain
        self.prompt_set = PromptSetRegistry.get(self.domain)
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        

    def _process_inputs(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str):
        system_prompt = f"{self.role}.\n{self.constraint}"
        decision_few_shot = self.prompt_set.get_decision_few_shot()
        history_info = ""
        for msg in history_messages:
            if msg.source == 'user':
                continue
            role_info = f" ({role_map[msg.source]})" if msg.source in role_map else ""
            history_info += f"{msg.source}{role_info}: {msg.content}\n\n"
        user_prompt = f"{decision_few_shot}\nThe task is:\n\n{task}\nAt the same time, the output of other agents is as follows:\n\n{history_info}"
        return system_prompt, user_prompt
    
    async def run_decision(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str) -> TextMessage:
        system_prompt, user_prompt = self._process_inputs(history_messages, role_map, task)


        completion = await tracked_create(
            self._model_client.chat.completions.create, stage="final_decision", source=self.name,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
 
        )
        response = completion.choices[0].message
        # response_message = TextMessage(content=response.content, source=self.name, models_usage=response.usage)
        response_message = TextMessage(content=response.content, source=self.name, models_usage=safe_request_usage(completion))
        
        return response_message
        