from masrubric.usage import tracked_create
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from typing import Dict
from autogen_core.models import UserMessage, ModelInfo
import re
from typing import List
from openai import AsyncOpenAI


SCORE_PROMPT = {
    "accuracy": (
        "Score the factual and task-specific correctness of the agent output on a 1-5 scale.\n"
        "Task:\n{task}\n\nAgent output:\n{agent_output}\n\n"
        "Return the score as '<Score> N'."
    ),
    "logical_soundness": (
        "Score whether the agent output follows a coherent and valid reasoning path on a 1-5 scale.\n"
        "Task:\n{task}\n\nAgent output:\n{agent_output}\n\n"
        "Return the score as '<Score> N'."
    ),
    "impactfulness": (
        "Score whether the output contributes useful progress toward solving the task on a 1-5 scale.\n"
        "Task:\n{task}\n\nAgent output:\n{agent_output}\n\n"
        "Return the score as '<Score> N'."
    ),
}

class Supervisor():
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        # role_map: Dict[str, str],
        metrics: list=["accuracy", "logical_soundness", "impactfulness"],
        weights: list[float]=[0.4, 0.4, 0.2],
        sample_times: int=3,
        threshold: float=3.0
    ):
        # self._model_client = OpenAIChatCompletionClient(
        #     model=model,
        #     api_key=api_key,
        #     base_url=base_url,
        #     temperature=0.0,
        #     model_info=ModelInfo(
        #         vision=False,
        #         function_calling=False,
        #         json_output=False,
        #         family="qwen3",
        #         structured_output=False
        #     )
        # )
        
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.metrics = metrics
        self.scoreboard: Dict[str, Dict[str, TextMessage | int]] = {}
        self.weights = weights
        # self.role_map = role_map
        self.sample_times = sample_times
        self.threshold = threshold
        
        if len(metrics) != len(weights):
            raise ValueError("Length of metrics and weights must be the same.")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.")
    
    def _parse_score(self, response: str) -> int:
        
   
        match = re.search(r'<Score>\s*(\d+)\s*$', response.strip(), re.MULTILINE)
        if match:
            return int(match.group(1))
        
        match_2 = re.search(r'</Score>\s*(\d+)\s*$', response.strip(), re.MULTILINE)
        if match_2:
            return int(match_2.group(1))
        raise ValueError(f"No valid score found in response: {response}")
    
    async def _calc_score(self, task, message: TextMessage) -> float:
        scores = {}
        for metric in self.metrics:
            prompt = SCORE_PROMPT[metric].format(
                task=task,
                agent_output=message.content
            )
            # input_messages = [UserMessage(content=prompt, source="user")]
            
            max_attempt = 10
            cur_attempt = 0
            current_scores = []
            while True:
                try:
                    # response = await self._model_client.create(
                    #     messages=input_messages,
                    # )
                    completion = await tracked_create(
                        self._model_client.chat.completions.create, stage="supervisor_audit", source="Supervisor", metadata={"audit_attempt": cur_attempt + 1},
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )
                    response = completion.choices[0].message
                    new_score = self._parse_score(response.content)
                    current_scores.append(new_score)
                    if len(current_scores) >= self.sample_times:
                        break
                except Exception as e:
                    cur_attempt += 1
                    if cur_attempt >= max_attempt:
                        raise e
                    print(f"Error in scoring with metric {metric}, retrying... ({cur_attempt}/{max_attempt})")
            # scores.append(sum(current_scores) / len(current_scores))
            scores[metric] = sum(current_scores) / len(current_scores)
        comprehensive_score = sum(scores[self.metrics[i]] * self.weights[i] for i in range(len(self.metrics)))
        return {**scores, "avg": comprehensive_score}
    
    async def update_scoreboard(self, task: str, message: TextMessage):
        if self.threshold == 0.0:
            self.scoreboard[message.id] = {
            "message": message,
            **{metric: None for metric in self.metrics},
            "avg": None
        }
        
        else:
            score = await self._calc_score(task, message)
            print(f"Message ID {message.id} scored: {score}")
            self.scoreboard[message.id] = {
                "message": message,
                **score
            }
        
    def prune_info(self, all_messages: List[TextMessage]) -> List[TextMessage]:
        if self.threshold == 0.0:
            return all_messages
        pruned_messages = []
        for msg in all_messages:
            if msg.id in self.scoreboard:
                if self.scoreboard[msg.id]["avg"] >= self.threshold:
                    pruned_messages.append(msg)
                else:
                    print(f"Info: Message ID {msg.id} filtered out by supervisor due to low score.")
            else:
                if msg.source != "user":
                    print(f"Warning: Message ID {msg.id} not found in scoreboard. Passing by default...")
                pruned_messages.append(msg)
        return pruned_messages
    
    def get_messages_above_threshold(self) -> List[TextMessage]:
        if self.threshold == 0.0:
            return [entry["message"] for entry in self.scoreboard.values()]
        return [entry["message"] for entry in self.scoreboard.values() if entry["avg"] >= self.threshold]
    
    def get_scores(self, role_map):
        # ret_scores = {}
        # for entry in self.scoreboard:
        #     ret_scores[entry] = {"role": role_map[self.scoreboard[entry]["message"].source], **{metric: self.scoreboard[entry][metric] for metric in self.metrics}, "avg": self.scoreboard[entry]["avg"]}
        # return ret_scores
        ret_scores = {}
        for entry in self.scoreboard:
            ret_scores[entry] = {
                'role': role_map[self.scoreboard[entry]["message"].source],
                'message': self.scoreboard[entry]['message'].dump(),
                'scores': {**{metric: self.scoreboard[entry][metric] for metric in self.metrics}, 'avg': self.scoreboard[entry]['avg']},
            }
        return ret_scores
    
    def reset(self):
        self.scoreboard = {}
