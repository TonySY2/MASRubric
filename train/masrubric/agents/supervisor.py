from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from typing import Dict, List, Tuple, Any, Optional, Union
from autogen_core.models import UserMessage, ModelInfo
import re
from openai import AsyncOpenAI
import json
import os
from json_repair import repair_json
import numpy as np
import traceback
import asyncio
import sys


TEACHER_B_MATH_PROMPT = """
You are an AI acting as a **Lead Mathematics Auditor and Logic Specialist**, specifically optimized for the MATH dataset (high-difficulty competitions like AMC, AIME).

### 📋 Background & Goal
**Background**: An agent team has attempted to solve a complex math problem, and **the team's final answer is INCORRECT**.
**Goal**: Synthesize the known problem (`problem`), standard solution (`solution`), and the Agent's `output` to strictly evaluate the Agent's reasoning process.

**IMPORTANT CONTEXT**: 
The provided `solution` is a standard, single-path reference answer. However, the Agent is part of a Multi-Agent System (MAS). 
- Its `output` depends on its `agent_role` (e.g., a "Python Coder" writes code, a "Critic" critiques). 
- **DO NOT** penalize the Agent simply because its output does not look like the standard `solution` (e.g., using code instead of pure derivation is valid if the role permits).
- Only penalize **logical errors**, **calculation errors**, or **hallucinations** that contradict mathematical truths.

### 📝 MATH Input Context
1. **`problem`**: 
{problem}

2. **`solution`**: (Ground Truth)
{solution}

3. **`agent_role`**: 
{agent_role}

4. **`output`**: (Agent's Attempt)
{output}

### 🧠 Phase 1: Diagnosis
Please execute the following logical judgment:
1. Assess whether the Agent's output is logically and mathematically correct **within the scope of its role**.
2. **AUDIT STRATEGY (CRITICAL)**:
   - **DO NOT STOP at the first error.** You must scan the ENTIRE output line by line.
   - Independent errors often exist (e.g., a logical fallacy in Step 1 AND a formatting error in the Final Answer).
   - You are expected to find **MULTIPLE distinct errors** (less than 5) if they exist.
3. **Decision**:
   - If the output contains **NO errors**: Output `NO_ERROR`.
   - If the output contains **errors**: Identify **ALL** of them and proceed to Phase 2.

### 🛠️ Phase 2: Metric Extraction
Transform **EACH identified error** separately into a **generalized** JSON metric object.
**CRITICAL**: The `name`, `detailed_definition`, `trigger_condition`, and `risk_alert` must be **generalizable** to other similar math problems.

1. **`specific_diagnostic_report`**:
    *   **Requirement**: **Verbatim quote** the exact part of the Agent's output where the error occurred, followed by an explanation of **why** it is wrong.
    *   **Format**: "Quote from output": Explanation of the error.

2. **`name`**:
    *   **Requirement**: Summarize the error pattern. It can be **appropriately longer** to avoid ID collisions.
    *   **Format**: `UPPER_CASE_WITH_UNDERSCORES`.

3. **`detailed_definition`**:
    *   **Requirement**: Define this error pattern conceptually. Describe the mathematical logical flaw in a way that applies to this category of math problems.

4. **`evaluator_prompt`**:
    Contains two key sub-fields to guide future audits:

    *   **`trigger_condition`**:
        *   **Thinking Process**: Based on this instance, under what general circumstances (problem type or agent behavior) is this error likely to occur?
        *   **Sentence Requirement**: Must start with **"When the problem involves [xx]..."** OR **"When the agent's output shows [xx]..."**. **At least one of these two is required.**

    *   **`risk_alert`**:
        *   **Thinking Process**: Provide a specific checklist item derived from this specific error point. Do not imagine unrelated checks; focus on preventing *this specific type* of mistake.
        *   **Sentence Requirement**: Must start with **"Attention! Check if..."**.

### 📤 Output Format
- If no error: Output `NO_ERROR` only.
- If errors exist: **ALWAYS Output a JSON LIST** containing one or more metric objects.
  - Structure: `[ {{ "name": "ERROR_1", ... }}, {{ "name": "ERROR_2", ... }} ]`
  - Even if there is only 1 error, wrap it in a list: `[ {{ ... }} ]`.
- **CRITICAL JSON SYNTAX RULE**:
  - When writing LaTeX inside JSON strings, **YOU MUST DOUBLE-ESCAPE BACKSLASHES**.
  - **WRONG**: `"equation": "\\frac{{1}}{{2}}"` (This causes JSON parse error!)
  - **CORRECT**: `"equation": "\\\\frac{{1}}{{2}}"` (This works!)
  - Or simply use plain text description to avoid issues (e.g., "fraction 1 over 2").
"""

# ==============================================================================
TEACHER_B_AQUA_PROMPT = """
You are an AI acting as a **Lead Algebra Auditor and Multiple-Choice Logic Specialist**, specifically optimized for the AQuA-RAT dataset (Algebra Question Answering with Rationales).

### 📋 Background & Goal
**Background**: An agent team has solved a math multiple-choice question, and **the team's final answer is INCORRECT**.
**Goal**: Synthesize the known problem (`question`), options (`options`), standard rationale (`rationale`), correct option (`correct`), and the Agent's `output` to strictly evaluate the Agent's reasoning process.

**IMPORTANT CONTEXT**: 
The Agent is part of a Multi-Agent System (MAS). 
- Its `output` depends on its `agent_role` (e.g., a "Math Solver" calculates, a "Programming Expert" codes).
- **DO NOT** penalize the Agent simply because its output does not look like the standard `rationale`.
- Only penalize **logical errors**, **calculation errors**, **hallucinations**, or **invalid option selection logic** that contradict mathematical truths.

### 📝 AQuA Input Context
1. **`question`**: 
{question}

2. **`options`**: 
{options}

3. **`rationale`**: (Ground Truth Reasoning)
{rationale}

4. **`correct`**: (Correct Option)
{correct}

5. **`agent_role`**: 
{role}

6. **`output`**: (Agent's Attempt)
{output}

### 🧠 Phase 1: Diagnosis
Please execute the following logical judgment:
1. Assess whether the Agent's output is logically and mathematically correct **within the scope of its role**.
2. **AUDIT STRATEGY (CRITICAL)**:
   - **DO NOT STOP at the first error.** You must scan the ENTIRE output line by line.
   - Independent errors often exist (e.g., a calculation error in Step 1 AND a logic flaw in option elimination).
   - You are expected to find **MULTIPLE distinct errors** (less than 5) if they exist.
3. **Decision**:
   - If the output contains **NO substantive errors**: Output `NO_ERROR`.
   - If the output contains **errors**: Identify **ALL** of them and proceed to Phase 2.

### 🛠️ Phase 2: Metric Extraction
Transform **EACH identified error** separately into a **generalized** JSON metric object.
**CRITICAL**: The `name`, `detailed_definition`, `trigger_condition`, and `risk_alert` must be **generalizable** to other similar algebraic word problems.

1. **`specific_diagnostic_report`**:
    *   **Requirement**: **Verbatim quote** the exact part of the Agent's output where the error occurred, followed by an explanation of **why** it is wrong based on the `rationale` and `correct` option.
    *   **Format**: "Quote from output": Explanation of the error.

2. **`name`**:
    *   **Requirement**: Summarize the error pattern. It can be **appropriately longer** to avoid ID collisions.
    *   **Format**: `UPPER_CASE_WITH_UNDERSCORES`.

3. **`detailed_definition`**:
    *   **Requirement**: Define this error pattern conceptually. Describe the mathematical logical flaw or option selection fallacy in a way that applies to this category of problems.

4. **`evaluator_prompt`**:
    Contains two key sub-fields to guide future audits:

    *   **`trigger_condition`**:
        *   **Thinking Process**: Based on this instance, under what general circumstances (problem type or agent behavior) is this error likely to occur?
        *   **Sentence Requirement**: Must start with **"When the problem involves [xx]..."** OR **"When the agent's output shows [xx]..."**. **At least one of these two is required.**

    *   **`risk_alert`**:
        *   **Thinking Process**: Provide a specific checklist item derived from this specific error point. Do not imagine unrelated checks; focus on preventing *this specific type* of mistake.
        *   **Sentence Requirement**: Must start with **"Attention! Check if..."**.

### 📤 Output Format
- If no error: Output `NO_ERROR` only.
- If errors exist: **ALWAYS Output a JSON LIST** containing one or more metric objects.
  - Structure: `[ {{ "name": "ERROR_1", ... }}, {{ "name": "ERROR_2", ... }} ]`
  - Even if there is only 1 error, wrap it in a list: `[ {{ ... }} ]`.
- **CRITICAL JSON SYNTAX RULE**:
  - When writing LaTeX inside JSON strings, **YOU MUST DOUBLE-ESCAPE BACKSLASHES**.
  - **WRONG**: `"equation": "\\frac{{1}}{{2}}"` (This causes JSON parse error!)
  - **CORRECT**: `"equation": "\\\\frac{{1}}{{2}}"` (This works!)
  - Or simply use plain text description to avoid issues (e.g., "fraction 1 over 2").
"""


class Supervisor():
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        embedding_model_name: str,
        embedding_api_key: str,
        embedding_base_url: str,
        domain: str = 'math', 
        supervisor_mode: str = 'collect',
        sample_times: int = 1,
        elite_strictness_coeff: float = 1.0, 
        warmup_rounds: int = 2               
    ):
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        

        self.domain = domain.lower() 
        
        self.supervisor_mode = supervisor_mode
        self.sample_times = sample_times
        
        self.embedding_model_name = embedding_model_name
        self._embedding_client = AsyncOpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
        
        self.elite_strictness_coeff = elite_strictness_coeff
        self.warmup_rounds = warmup_rounds

        self.scoreboard = {}
        self.trajectories = {} 
        self.global_trajectory_log = []
        self.current_full_task_json = None
        
    def _parse_json_from_response(self, response: str) -> List[Dict]:
        response = response.strip()
        
        # 1. 尝试使用正则提取 Markdown 代码块中的 JSON
        # 使用 re.DOTALL 让 . 匹配换行符
        match_list = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL | re.IGNORECASE)
        if match_list:
            json_str = match_list.group(1)
            try:
                parsed = repair_json(json_str, return_objects=True)
                if isinstance(parsed, list): return parsed
            except: pass

        # 2. 如果正则没找到，尝试暴力寻找最外层的 [ ]
        # 很多时候模型会在最后输出 JSON，我们找最后一个 ] 和对应的 [
        try:
            # 从后往前找最后一个 ']'
            end = response.rfind(']')
            if end != -1:
                # 从开始找第一个 '['
                start = response.find('[')
                if start != -1 and end > start:
                    potential_json = response[start : end+1]
                    parsed = repair_json(potential_json, return_objects=True)
                    if isinstance(parsed, list): return parsed
        except: pass

        # 3. 兜底：如果上面都失败，尝试直接修复整个字符串
        try:
            parsed = repair_json(response, return_objects=True)
            if isinstance(parsed, list): return parsed
            if isinstance(parsed, dict): return [parsed]
        except: pass

        print(f"[Supervisor] ⚠️ JSON Parse Failed. Raw content start: {response[:100]}...")
        # 返回空列表而不是包含 error 的 dict，避免后续流程报错，或者保留你的 error 逻辑
        return [{"error": "JSON parsing failed", "raw": response}]
    # ==============================================================================

    async def _generate_adversarial_metrics(self, trajectory: Dict[str, Any], full_task_json: str) -> List[Dict[str, Any]]:
        if not full_task_json:
            return []

        try:
            task_data = json.loads(full_task_json)
            
            agent_name = trajectory.get("agent_name", "Unknown")
            agent_role = trajectory.get("agent_role", "")
            agent_output = trajectory.get("output", "")
            agent_desc = trajectory.get("agent_role_description", "")
            role_input = f"{agent_role}: {agent_desc}" if agent_desc else agent_role
            
            b_prompt = ""
            metadata_template = {}

      
            if "math" in self.domain:
       
                problem_text = task_data.get("problem", task_data.get("question", ""))
                solution_text = task_data.get("solution", task_data.get("answer", "")) 
                
                b_prompt = TEACHER_B_MATH_PROMPT.format(
                    problem=problem_text,
                    solution=solution_text,
                    agent_role=role_input,
                    output=agent_output
                )
                
                metadata_template = {
                    "source_task_id": task_data.get("id", "unknown"),
                    "source_dataset": "math",
                    "ground_truth_solution": solution_text
                }

            elif "aqua" in self.domain:

                question_text = task_data.get("question", "")
                options_list = task_data.get("options", [])
                rationale_text = task_data.get("rationale", "")
                correct_option = task_data.get("correct", "")
                
                b_prompt = TEACHER_B_AQUA_PROMPT.format(
                    question=question_text,
                    options=str(options_list), 
                    rationale=rationale_text,
                    correct=correct_option,
                    role=role_input,
                    output=agent_output
                )
                
                metadata_template = {
                    "source_task_id": task_data.get("id", "unknown"),
                    "source_dataset": "aqua",
                    "ground_truth_rationale": rationale_text,
                    "correct_option": correct_option
                }
            else:
        
                print(f"[Supervisor] ⚠️ Warning: Unknown domain '{self.domain}', using default MATH logic.")
                problem_text = task_data.get("problem", task_data.get("question", ""))
                solution_text = task_data.get("solution", task_data.get("answer", ""))
                b_prompt = TEACHER_B_MATH_PROMPT.format(
                    problem=problem_text, solution=solution_text, agent_role=role_input, output=agent_output
                )
                metadata_template = {"source_dataset": "unknown_fallback"}


            resp_b = await self._model_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": b_prompt}],
                temperature=0.0,
       
                frequency_penalty=0.5, 
                presence_penalty=0.2
            )
            raw_content_b = resp_b.choices[0].message.content.strip()
            

            print(f"\n{'='*20} [DEBUG TEACHER B RAW ({self.domain})] {'='*20}")
            print(raw_content_b)
            print(f"{'='*50}\n")
            
            if "NO_ERROR" in raw_content_b:
                print(f"[Supervisor] ⚪ Teacher B found NO logic error.")
                return []

            candidates_data_list = self._parse_json_from_response(raw_content_b)
            final_accepted_metrics = []


            for idx, generated_data in enumerate(candidates_data_list):
                
   
                if "error" in generated_data and "raw" in generated_data:
                    print(f"[Supervisor] 🔴 Item {idx+1}: JSON Parse Error.")
                    continue
                
 
                # === 新增：键名归一化处理 ===
                # 将所有键转换为小写，处理 'Name', 'NAME', 'Error_Name' 等情况
                normalized_data = {}
                for k, v in generated_data.items():
                    normalized_data[k.lower()] = v
                
                # 尝试多种可能的 key 获取 name
                metric_name = normalized_data.get("name") or \
                              normalized_data.get("error_name") or \
                              normalized_data.get("error_type") or \
                              ""
                metric_name = metric_name.strip()
                # ===========================
                
                
                if not metric_name:
           
                    print(f"[Supervisor] ⚠️ Item {idx+1}: Missing 'name'. Auto-fixing...")
                    metric_name = "UNNAMED_LOGIC_ERROR"
                
   
                if "NO_ERROR" in metric_name.upper() or "NO_SUBSTANTIVE" in metric_name.upper():
                    print(f"[Supervisor] ⚪ Ignored 'No Error' JSON: {metric_name}")
                    continue

         
                final_metric = {
                    "specific_diagnostic_report": generated_data.get("specific_diagnostic_report", "N/A"),
                    "name": metric_name,
                    "detailed_definition": generated_data.get("detailed_definition", "No definition provided."),
                    "evaluator_prompt": generated_data.get("evaluator_prompt", {}),
                    "metadata": metadata_template
                }
                
 
                final_accepted_metrics.append(final_metric)
                print(f"[Supervisor] 🟢 Item {idx+1}: Raw Metric Generated: {final_metric['name']}")

            return final_accepted_metrics

        except Exception as e:
            print(f"[Supervisor ERROR] Metric generation failed: {e}")
            traceback.print_exc()
            return []

    async def collect_trajectory(self, trajectory: Dict[str, Any], message: TextMessage, full_task_json: str = ""):
        if self.supervisor_mode != 'collect':
            return
        
        if message and hasattr(message, 'id'):
            self.scoreboard[message.id] = {"message": message}
            
        agent_name = trajectory.get("agent_name")
        if not agent_name:
            return

        trajectory['full_task_json'] = full_task_json
        trajectory['generated_adversarial_metrics'] = []
        
        if agent_name not in self.trajectories:
            self.trajectories[agent_name] = []
        self.trajectories[agent_name].append(trajectory)
        
        self.global_trajectory_log.append(trajectory)

    async def review_failed_trajectories(self):
        print(f"[Supervisor] 🔍 Start auditing {len(self.global_trajectory_log)} steps for FAILED task...")
        
        for trajectory in self.global_trajectory_log:
            full_task_json = trajectory.get('full_task_json', "")
            metrics = await self._generate_adversarial_metrics(trajectory, full_task_json)
            trajectory['generated_adversarial_metrics'] = metrics
            
        return self.trajectories

    def get_collected_trajectories(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.trajectories

    def prune_info(self, all_messages: List[TextMessage]) -> List[TextMessage]:
        return all_messages
    
    def get_messages_above_threshold(self) -> List[TextMessage]:
        all_scored_messages = [entry["message"] for entry in self.scoreboard.values()]
        return self.prune_info(all_scored_messages)
    
    def reset(self):
        self.scoreboard = {}
        self.trajectories = {}
        self.global_trajectory_log = []
        self.current_full_task_json = None