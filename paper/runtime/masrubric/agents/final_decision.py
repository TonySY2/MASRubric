from masrubric.agents.agent_registry import AgentRegistry
from masrubric.prompt.prompt_set_registry import PromptSetRegistry
from masrubric.usage_tracking import record_chat_completion
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage, SystemMessage, RequestUsage
from typing import List, Dict
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from masrubric.tools.coding.python_executor import PyExecutor
#from transformers import AutoTokenizer
from masrubric.prompt.mbpp_prompt_set import FEW_SHOT_DATA_MBPP



@AgentRegistry.register('FinalWriteCodeMBPP')
class FinalWriteCodeMBPP:
    def __init__(self, name: str, model: str, api_key: str, base_url: str, domain: str):
        self._model_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )
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
    
    # async def run_decision(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str) -> TextMessage:
    #     system_prompt, user_prompt = self._process_inputs(history_messages, role_map, task)
    #     # [开始修改位置] ------------------------------------------------
        
    #     # 1. 基础 System Message
    #     messages_to_send = [{"role": "system", "content": system_prompt}]
        
    #     # 2. [核心修改] 注入 Few-Shot
    #     # 这里我们明确指定使用 "DecisionMaker" 的示例
    #     decision_role_key = "DecisionMaker"
    #     if decision_role_key in FEW_SHOT_DATA_MBPP:
    #         for example_q, example_a in FEW_SHOT_DATA_MBPP[decision_role_key]:
    #             messages_to_send.append({"role": "user", "content": example_q})
    #             messages_to_send.append({"role": "assistant", "content": example_a})
                
    #     # 3. 放入当前轮次的 User Prompt
    #     messages_to_send.append({"role": "user", "content": user_prompt})

    #     completion = await self._model_client.chat.completions.create(
    #         model=self.model,
    #         messages=messages_to_send, # <--- 替换这里
    #         temperature=0.0, # 决策者应该更确定
    #     )
    #     # [修改结束] ------------------------------------------------
    #     response = completion.choices[0].message
    #     usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)
    #     response_message = TextMessage(content=response.content, source=self.name, models_usage=usage)
    #     return response_message
    
    async def run_decision(self, history_messages: List[TextMessage], role_map: Dict[str, str], task: str) -> TextMessage:
        system_prompt, user_prompt = self._process_inputs(history_messages, role_map, task)
        completion = await self._model_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # 决策者应该更确定
        )
        record_chat_completion(completion, model=self.model, stage="final_decision", source=self.name)
        response = completion.choices[0].message
        usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)
        response_message = TextMessage(content=response.content, source=self.name, models_usage=usage)
        return response_message

    
    
@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode:
    def __init__(self, name: str, model: str, api_key: str, base_url: str, domain: str):
        self._model_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )
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
        completion = await self._model_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # 决策者应该更确定
        )
        record_chat_completion(completion, model=self.model, stage="final_decision", source=self.name)
        response = completion.choices[0].message
        usage = RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens)
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
        # self._model_client = OpenAIChatCompletionClient(
        #     model=model,
        #     api_key=api_key,
        #     base_url=base_url,
        #     temperature=0.0,
        # )

        self._model_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )
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
        # input_messages = [
        #     SystemMessage(content=system_prompt),
        #     UserMessage(content=user_prompt, source="user")
        # ]
        
        # response = await self._model_client.create(
        #     messages=input_messages,
        # )
        #tokenizer = AutoTokenizer.from_pretrained("<historical-path>")
        completion = await self._model_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            # extra_body={
            #     # "stop": ["<|eot_id|>", "</s>", "<|im_end|>"],
            #     "stop_token_ids":[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            #     "chat_template_kwargs": {"enable_thinking": False}}
        )
        record_chat_completion(completion, model=self.model, stage="final_decision", source=self.name)
        response = completion.choices[0].message
        # response_message = TextMessage(content=response.content, source=self.name, models_usage=response.usage)
        response_message = TextMessage(content=response.content, source=self.name, models_usage=RequestUsage(prompt_tokens=completion.usage.prompt_tokens, completion_tokens=completion.usage.completion_tokens))
        
        return response_message
        
