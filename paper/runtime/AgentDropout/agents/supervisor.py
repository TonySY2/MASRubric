from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from typing import Dict
from autogen_core.models import UserMessage, ModelInfo
import re
from typing import List
from openai import AsyncOpenAI


SCORE_PROMPT = {
"accuracy":
"""You are a high-level Accuracy Scorer. Your task is to judge the correctness and fidelity of an agent's response by applying the strict scoring rubric provided below.

## Overall Task:
{task}

## Agent Output to be Evaluated:
{agent_output}

## Judgment Instructions:
Your goal is to score the agent's output from 1 to 100. A highly accurate output must clearly demonstrate a thorough understanding of the task, correctly interpret the evidence and reasoning, and provide a clear, consistent rationale for its final answer. Your judgment must strictly adhere to, accurately apply, and remain consistent with these instructions.

## Scoring Rubric:

- **Score 81 to 100 (Perfectly Accurate):** The output perfectly demonstrates a comprehensive and nuanced understanding of the task. It flawlessly and with high consistency applies all reasoning criteria. The final answer is perfectly correct, and the explanation is clear, detailed, and precisely articulates how the evidence leads to the correct conclusion.

- **Score 61 to 80 (Good Accuracy):** The output shows a good understanding of the task and, in most cases, applies reasoning accurately and consistently. There might be slight misinterpretations or deviations in its analysis, but these minor issues do not significantly affect the correctness of the final answer. The rationale is clear and largely supports the decision.

- **Score 41 to 60 (Partial Accuracy):** The output shows a basic understanding of the task, but there are some noticeable inconsistencies or omissions in its reasoning. Some parts of the analysis may be correct, but other key parts were ignored or misinterpreted, leading to a final answer that is only partially accurate or correct for the wrong reasons.

- **Score 21 to 40 (Severe Inaccuracy):** The output shows a severe deficit in understanding the task. Multiple major errors were made in its reasoning, leading to a clearly inaccurate conclusion. The rationale shows significant inconsistency and fails to effectively support the decision.

- **Score 1 to 20 (Completely Inaccurate):** The output completely ignores or severely misinterprets the task. The final answer appears arbitrary and is entirely disconnected from a logical reasoning process. Its explanation (if provided) contradicts the facts or is irrelevant.

## Your Task:
Provide your evaluation and a brief rationale based on the rubric above. Your response must end with the score in the specified format.

Your Response (Rationale and Score):

<Rationale>
[Your brief reason]

<Score>
[From 1 to 100]
""",

"logical_soundness":
"""You are a a specialist in logical reasoning. Your task is to assess whether an agent's reasoning follows a coherent and logical progression0.

## Overall Task:
{task}

## Agent Output to be Evaluated:
{agent_output}

## Judgment Criterion: Logical Soundness
Your goal is to score the agent's output from 1 to 100. A well-reasoned decision should clearly demonstrate how conclusions were drawn and avoid logical fallacies or contradictions. This ensures the reasoning process is transparent and defensible.

## Scoring Rubric:

- **Score 81 to 100 (Entirely Logical):** The decision-making process is entirely logical, with clear and consistent reasoning throughout. Every step in the reasoning process is well-supported and leads naturally to the conclusion.

- **Score 61 to 80 (Mostly Logical):** The decision-making process is mostly logical, with minor issues that do not undermine its overall integrity. The reasoning is generally clear and follows a structured progression with only slight missteps.

- **Score 41 to 60 (Moderately Logical):** The decision-making process is moderately logical, but some inconsistencies or gaps weaken its coherence. While the reasoning is partially sound, certain steps may appear unclear or unsupported.

- **Score 21 to 40 (Significant Logical Flaws):** The decision-making process shows significant gaps or logical flaws, making it difficult to follow. Reasoning is inconsistent, and critical errors undermine the validity of the conclusion.

- **Score 1 to 20 (Illogical):** The decision-making process is illogical, lacking clear reasoning or consistency. The conclusion appears arbitrary or disconnected from the supporting evidence.

## Your Task:
Provide your evaluation and a brief rationale based on the rubric above. Your response must end with the score in the specified format.

Your Response (Rationale and Score):

<Rationale>
[Your brief reason]

<Score>
[From 1 to 100]
""",

"impactfulness":
"""You are an expert in multi-agent system dynamics. Your task is to assess the value and effectiveness of information passed from one agent to another, in terms of advancing task progress.

## Overall Task:
{task}

## Agent Output to be Evaluated:
{agent_output}

## Judgment Criterion: Impactfulness (Information Impact)
Your goal is to score the agent's output from 1 to 100. A highly impactful piece of information should be clear, actionable, and provide the receiving agent with the critical data or instructions necessary to complete its sub-task. It should enhance the efficiency or quality of subsequent steps.

## Scoring Rubric:

- **Score 81-10 (Critical/Decisive Impact):** The information is exceptionally concise, clear, and provides critical, irreplaceable insights or instructions. It is a decisive factor in the success of the entire task chain, potentially unlocking a more optimal solution or preventing a critical failure.

- **Score 61-80 (Significant Impact):** The information is clear, accurate, and directly tailored to the needs of the receiving agent. It significantly improves the efficiency or quality of subsequent tasks.

- **Score 41-60 (Moderate Impact):** The information is relevant and useful, allowing the receiving agent to execute its tasks reasonably well. However, it may not be sufficiently concise or may be slightly lacking in clarity. It serves as a standard contribution.

- **Score 21-40 (Limited Impact):** The information has some relevance but lacks key details or clear instructions. The receiving agent needs to perform significant extra inference to make use of it. Its contribution is negligible.

- **Score 1-20 (Low/Negative Impact):** The information is irrelevant, inaccurate, or extremely ambiguous, hindering the overall task workflow.

## Your Task:
Evaluate the "Agent Output" based on its potential impact on a receiving agent. Provide your evaluation and a brief rationale. Your response must end with the score in the specified format.

Your Response (Rationale and Score):

<Rationale>
[Your brief reason]

<Score>
[From 1 to 100]
"""
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
        
        # 从末尾开始匹配 <Score> 标签后的数字
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
                    completion = await self._model_client.chat.completions.create(
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
