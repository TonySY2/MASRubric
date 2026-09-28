#--- START OF FILE supervisor_reasoning_pick_metric.py ---

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from typing import Any, Dict
from autogen_core.models import UserMessage, ModelInfo
import re
from typing import List
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient
import json
import threading
from openai import OpenAI
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from json_repair import repair_json
import os
import random  # <--- 加上这个
from datetime import datetime, timezone
from pathlib import Path
from masrubric.usage_tracking import record_chat_completion, record_embedding_response
from masrubric.profileaware_metrics import (
    MATCH_PROFILE_SCHEMA,
    build_metric_embedding_text,
    build_query_embedding_text,
    has_profile_conflict,
    infer_profile_heuristically,
    normalize_match_profile,
)


# -------------------------------------------------------------------------
# [新增] 固定的通用审计指标 (Simple Audit Mode)
# 这是一个“虚拟指标”，伪装成 RAG 检索回来的样子，但内容是通用的
# -------------------------------------------------------------------------
SIMPLE_MATH_METRIC_1 = {
    "name": "GENERAL_LOGIC_AND_MATH_CHECK",
    "detailed_definition": "A comprehensive audit of the agent's mathematical derivation, logic consistency, and constraint satisfaction.",
    "evaluator_prompt": {
        # Trigger 设置为总是触发
        "trigger_condition": "The Agent is performing mathematical reasoning to solve a problem.",
        # Risk Alert 设置为通用的逻辑检查指令
        "risk_alert": "Attention! Check for ANY fatal logic gaps, arithmetic hallucinations, or misinterpretation of problem constraints. Focus only on errors that impact the final result."
        "Be careful not to ask this agent to do things outside of their responsibilities, just to see if what they are doing is right or wrong"
    }
}


# -------------------------------------------------------------------------
# [新增] 固定的通用代码审计指标 (Simple Audit Mode - For Code)
# 适用于 MBPP, HumanEval 等代码生成任务
# -------------------------------------------------------------------------
SIMPLE_CODE_METRIC_1 = {
    "name": "GENERAL_CODE_CORRECTNESS_CHECK",
    "detailed_definition": "A comprehensive audit of the agent's code implementation, focusing on functional correctness, logic integrity, syntax validity, and adherence to problem constraints.",
    "evaluator_prompt": {
        # Trigger 设置为总是触发
        "trigger_condition": "The Agent is generating, analyzing, or debugging computer code.",
        # Risk Alert 专注于代码运行错误、逻辑漏洞和API幻觉
        "risk_alert": "Attention! Check for ANY fatal syntax errors, logic bugs (e.g., infinite loops, off-by-one), library hallucinations, or interface mismatches. Focus only on errors that cause runtime failures or incorrect outputs."
        "Be careful not to ask this agent to do things outside of their responsibilities, just to see if what they are doing is right or wrong"
    }
}




# ==============================================================================
# [新增] Random/Ablation 专用模板：
# ==============================================================================

METRIC_TEMPLATE_RANDOM_MATH = """
You are an Objective Logic Auditor.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL LOGIC ERROR** regarding a specific **Area of Concern**.

### 🛑 Relevance Pre-Check (CRITICAL)
Before auditing, you must strictly evaluate if the **[Area of Concern]** is actually relevant to the current Task and Agent Output.
- **If Irrelevant**: (e.g., the metric checks "Probability" but the task is "Geometry"), you must **STOP** and PASS the agent. In the JSON, write "Metric not applicable" in `analysis`, "N/A" in `suggestion`, and set `is_flawed` to `false`.
- **If Relevant**: Proceed to the Impact & Action Protocol below.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's reasoning is correct unless you find irrefutable evidence of a fatal flaw.
2. **The "Actionability" Test**: If you cannot provide a specific, mathematical correction (a formula, a step, or a value), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the Agent's phrasing is imperfect but the **FINAL ANSWER** remains mathematically correct, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic part. Write 'N/A' if valid or irrelevant.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on logic, not style. Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix it (e.g., 'Change x to y', 'Apply formula Z'). If no fix is needed or possible (or metric is irrelevant), write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does the FINAL ANSWER or core conclusion change? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""

METRIC_TEMPLATE_RANDOM_CODE = """
You are a Senior Code Auditor and Architect.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL CODING ERROR** regarding a specific **Area of Concern**.

### 🛑 Relevance Pre-Check (CRITICAL)
Before auditing, you must strictly evaluate if the **[Area of Concern]** is technically applicable to the current Code.
- **If Irrelevant**: (e.g., the metric checks "Database" but the code is "Sorting Array"), you must **STOP** and PASS the agent. In the JSON, write "Metric not applicable" in `analysis`, "N/A" in `suggestion`, and set `is_flawed` to `false`.
- **If Relevant**: Proceed to the Impact & Action Protocol below.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's code is functionally correct unless you find irrefutable evidence of a fatal flaw (syntax error, logic bug, or interface violation).
2. **The "Actionability" Test**: If you cannot provide a specific code correction (a line change, a logic fix, or a parameter adjustment), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the code is inefficient, verbose, or stylistically non-standard but **EXECUTES CORRECTLY** and returns the right result, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic code snippet. Write 'N/A' if valid or irrelevant.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on functional correctness (bugs/crashes), not style (PEP8/comments).Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix the code (e.g., 'Change index i to i+1', 'Import module X'). If no fix is needed (or metric is irrelevant), write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does it fix a runtime error, infinite loop, or incorrect output? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""




# ==============================================================================
# [优化版] Simple Audit Metrics (Static)
# ==============================================================================

# 1. 数学/奥赛专用静态指标
SIMPLE_MATH_METRIC_2 = {
    "name": "CRITICAL_MATH_LOGIC_AUDIT",
    "detailed_definition": "A focused audit to detect substantive logical fallacies, calculation errors, or conditional oversights that invalidate the final result.",
    "evaluator_prompt": {
        # Trigger: 只要涉及数学推理就触发
        "trigger_condition": "The Agent is performing mathematical reasoning, derivation, or calculation.",
        
        # Risk Alert: 强调“审计员职责”和“只抓实锤错误”
        "risk_alert": (
            "You are an Objective Math Auditor. Your duty is to **verify** the agent's logic, not to rewrite their solution.\n"
            "**Audit Standards:**\n"
            "1. **Fatal Errors ONLY**: Flag specific steps that are mathematically FALSE. Do not critique efficiency, style, or 'better methods'. If the logic holds, let it pass.\n"
            "2. **Verify, Don't Assume**: Don't just look at the answer. Check if the intermediate deductions actually support the conclusion.\n\n"
            "**Potential Risk Areas to Scan (Heuristics):**\n"
            "- **Hallucinations**: Using non-existent theorems or making up numbers.\n"
            "- **Logic Gaps**: Jumping to conclusions without proof (e.g., assuming symmetry/maximums).\n"
            "- **Boundary Neglect**: Missing edge cases (zero, negative, empty sets) or necessary/sufficient conditions.\n"
            "- **Calculation Failures**: Basic arithmetic errors that propagate to the final result.\n"
            "3. Be careful not to ask this agent to do things outside of their responsibilities, just to see if what they are doing is right or wrong"
        )
    }
}

# 2. 代码/编程专用静态指标
SIMPLE_CODE_METRIC_2 = {
    "name": "CRITICAL_CODE_CORRECTNESS_CHECK",
    "detailed_definition": "A functional audit focusing on runtime safety, logical integrity, and adherence to requirements in code implementation.",
    "evaluator_prompt": {
        # Trigger: 只要涉及写代码就触发
        "trigger_condition": "The Agent is generating, debugging, or analyzing computer code.",
        
        # Risk Alert: 强调“运行安全”和“逻辑正确”，忽略风格
        "risk_alert": (
            "You are a Senior Code Auditor. Your specific goal is to catch bugs that would cause the code to **crash** or produce **wrong outputs**.\n"
            "**Audit Standards:**\n"
            "1. **Functionality First**: Ignore PEP8, variable naming, or comments unless they obscure logic. Focus on: Will this code run? Will it output the right thing?\n"
            "2. **Logic over Syntax**: While syntax errors are bad, subtle logic bugs (that run but give wrong answers) are worse.\n\n"
            "**Potential Risk Areas to Scan (Heuristics):**\n"
            "- **Runtime Hazards**: Infinite loops, recursion without base cases, division by zero, index out of bounds.\n"
            "- **Library Misuse**: Hallucinated methods or incorrect API usage.\n"
            "- **Edge Cases**: Handling of empty inputs, 0, 1, or extreme values.\n"
            "- **Logic Bugs**: Off-by-one errors, incorrect conditional branches, or variable shadowing."
            "3. Be careful not to ask this agent to do things outside of their responsibilities, just to see if what they are doing is right or wrong"
        )
    }
}

# --- [核心修改：新的开放式诊断模板] ---
METRIC_TEMPLATE_SINGLE_AUDIT = """
You are an Objective Logic Auditor.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL LOGIC ERROR** regarding a specific **Area of Concern**.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's reasoning is correct unless you find irrefutable evidence of a fatal flaw.
2. **The "Actionability" Test**: If you cannot provide a specific, mathematical correction (a formula, a step, or a value), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the Agent's phrasing is imperfect but the **FINAL ANSWER** remains mathematically correct, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic part. Write 'N/A' if valid.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on logic, not style. Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix it (e.g., 'Change x to y', 'Apply formula Z'). If no fix is needed or possible, write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does the FINAL ANSWER or core conclusion change? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""

METRIC_TEMPLATE_CODE_AUDIT = """
You are a Senior Code Auditor and Architect.
Your task is to verify if a specific team member (**Agent Role**) has committed a **FATAL CODING ERROR** regarding a specific **Area of Concern**.

### 🛡️ The "Impact & Action" Protocol
1. **Presumption of Validity**: You must assume the Agent's code is functionally correct unless you find irrefutable evidence of a fatal flaw (syntax error, logic bug, or interface violation).
2. **The "Actionability" Test**: If you cannot provide a specific code correction (a line change, a logic fix, or a parameter adjustment), **IT IS NOT A FLAW**.
3. **The "Impact" Test**: If the code is inefficient, verbose, or stylistically non-standard but **EXECUTES CORRECTLY** and returns the right result, **IT IS NOT A FLAW**.

### ⚖️ Judgment Criteria
**[Area of Concern]**: {trigger_condition}

---

### CONTEXT
- **Task**: {task}
- **Agent Role**: {role}
- **Agent Output**: {agent_output}

---

### OUTPUT FORMAT (JSON ONLY)
You must generate the fields in this **EXACT ORDER**. The logical flow determines the verdict.

{{
    "evidence_quote": "Verbatim quote of the problematic code snippet. Write 'N/A' if valid.",
    "analysis": "Explain WHY this specific part violates the Area of Concern. Focus on functional correctness (bugs/crashes), not style (PEP8/comments).Try to express in a concise and to the point manner, avoid lengthy speeches. Write 'N/A' if valid.",
    "suggestion": "Concrete instruction on how to fix the code (e.g., 'Change index i to i+1', 'Import module X'). If no fix is needed, write 'N/A'.",
    "impact_assessment": "Simulate the correction. Does it fix a runtime error, infinite loop, or incorrect output? (YES/NO) and brief reason.",
    "is_flawed": boolean // Set to true ONLY if 'suggestion' is concrete AND 'impact_assessment' is YES. Otherwise false.
}}
"""

METRIC_TEMPLATE_BATCH_AUDIT_MATH = """
You are an Objective Logic Auditor.
Evaluate the agent output against every metric in the Metric Library below.

### Core Rules
1. Judge each metric independently.
2. Assume the agent is correct unless you find a specific fatal error.
3. If you cannot give a concrete correction, mark `is_flawed` as false.
4. If the issue does not change the final answer or core conclusion, mark `is_flawed` as false.
5. You must return one result for every metric, in the same order, with the exact metric name.

### Context
- Task: {task}
- Agent Role: {role}
- Agent Output: {agent_output}

### Metric Library
{metrics_block}

### Output Format
Return a single valid JSON object:
{{
  "judgements": [
    {{
      "metric": "EXACT_METRIC_NAME",
      "evidence_quote": "Verbatim problematic span or N/A",
      "analysis": "Why this metric is violated, or N/A",
      "suggestion": "Concrete fix, or N/A",
      "impact_assessment": "YES/NO with a short reason",
      "is_flawed": false
    }}
  ]
}}
"""

METRIC_TEMPLATE_BATCH_AUDIT_CODE = """
You are a Senior Code Auditor.
Evaluate the agent output against every metric in the Metric Library below.

### Core Rules
1. Judge each metric independently.
2. Focus on bugs that cause crashes, wrong outputs, or requirement violations.
3. Ignore style-only issues.
4. If you cannot give a concrete code fix, mark `is_flawed` as false.
5. You must return one result for every metric, in the same order, with the exact metric name.

### Context
- Task: {task}
- Agent Role: {role}
- Agent Output: {agent_output}

### Metric Library
{metrics_block}

### Output Format
Return a single valid JSON object:
{{
  "judgements": [
    {{
      "metric": "EXACT_METRIC_NAME",
      "evidence_quote": "Verbatim problematic span or N/A",
      "analysis": "Why this metric is violated, or N/A",
      "suggestion": "Concrete fix, or N/A",
      "impact_assessment": "YES/NO with a short reason",
      "is_flawed": false
    }}
  ]
}}
"""





SUMMARY_TEMPLATE = """[Instruction]
Analyze the provided Task and Agent Output to extract key features for metric retrieval.
Strictly respond with a JSON object.

### Context
Task: {task}
Agent Output: {agent_output}

### Output JSON Format
{{
    "problem_scenario": ["Keyword 1", "Keyword 2"],
    "agent_action": ["Action Keyword 1", "Action Keyword 2"]
}}
"""

MATCH_PROFILE_TEMPLATE = """You generate a compact JSON match profile for code metric retrieval.

Rules:
1. Use only evidence visible in the task context and the current agent output.
2. Keep the schema exactly as requested.
3. Pick one value for interface_shape from: function_api, class_api, stdin_stdout, file_or_env, mixed, unknown.
4. Pick one value for io_contract from: return_value, stdout, mutation, exception, file_io, dependency, unknown.
5. match_hint must be a short paragraph that states the current context, visible implementation evidence, and the highest-value risk to match.
6. Do not mention benchmark names, agent roles, or hidden tests.
7. Keep the whole response within the JSON object.

Task Context:
{task}

Agent Output:
{agent_output}

Return JSON only:
{{
  "interface_shape": "unknown",
  "io_contract": "unknown",
  "match_hint": "..."
}}
"""

# --- [核心修改] 增强选择器 Prompt ---
SELECTION_TEMPLATE = """You are a Lead Logic Auditor. Your task is to select {selection_count_instruction} relevant metrics to evaluate an agent's output.

### INSTRUCTIONS:
1. **Analyze & Summarize**: First, summarize the task's core logic and the agent's specific mathematical/logical/code operations in the output.
2. **Scan Metrics**: Review the "Metric Library" provided below. Focus on the 'Trigger Condition' and 'Detailed Definition'.
3. **Match**: Select metrics that are directly applicable to the agent's specific actions or the problem's scenario.
4. **Output**: You MUST respond with a single, valid JSON object containing your summary and the list of selected metric names.

### Task
{task}

### Agent's Output
{agent_output}

### Metric Library (Pool)
{metrics_list}

### STRICT FINAL INSTRUCTION
Your final output MUST be a single valid JSON object. Do not add any preamble, conclusion, or markdown formatting.

### Expected JSON Output Format:
{{
    "summary": "Brief analysis of the agent's logic and potential risk areas...",
    "selected_metrics": ["METRIC_NAME_1", "METRIC_NAME_2"...]
}}
"""

CHEAP_PRECHECK_TEMPLATE = """You are a lightweight audit triage model.
Decide whether the following agent output may contain a substantive correctness problem.

Return exactly one word:
CHECK
or
PASS

Use CHECK if the output may contain a meaningful reasoning error, missed constraint, inconsistent answer, or code bug.
Use PASS if the output looks acceptable and does not need a full audit.

Task:
{task}

Agent Role: {role}
Agent Output:
{agent_output}
"""


CHEAP_PRECHECK_SKIP_ROLES = {"PROGRAMMING EXPERT"}
CHEAP_PRECHECK_TRACE_ENV = "MASRUBRIC_CHEAP_PRECHECK_TRACE"
_CHEAP_PRECHECK_TRACE_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class Supervisor():
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        domain: str,  
        
        # --- [新参数组：检索策略控制] ---
        force_direct_search: bool = False, # 是否强制直接检索（跳过 LLM Rerank）
        direct_k: int = 5,                 # 直选模式下的召回数量
        retrieve_p: int = 20,              # Rerank 模式下的粗排召回数量 (Candidate Pool)
        select_q: int = 5,                 # Rerank 模式下 LLM 最终精选数量 (Final Selection)
        # -------------------------------
        # [新增] 随机对照参数，默认为 0
        random_k: int = 0, 
        
        sample_times: int = 3,
        pass_rate: float = 1,
        prune_flag: bool = True,
        metric_pool_file: str = "",
        embedding_cache_file: str = "",
        embedding_model: str = "",
        embedding_api_key: str = "",
        embedding_api_base: str = "",
        # [新增参数] 接收预加载的数据
        preloaded_metrics: List[Dict] = None,
        
        #preloaded_embeddings 里装的是 “指标触发条件”（Trigger Condition） 的向量表示。 
        preloaded_embeddings: np.ndarray = None,
        
        # [新增] 接收开关
        lock_metrics_after_first_round: bool = False,
        
        # [新增参数] 简单审计模式开关
        use_simple_audit: int = 0, # [修改] 类型提示改为 int
        
        # [兼容性保留] 
        metrics_retrieve_k: int = 20, # 旧参数，仅作占位兼容
        use_llm_rerank: bool = True,  # 旧参数，仅作占位兼容
        max_metrics_count: int = 5    # 旧参数，仅作占位兼容
    ):
        self.domain = domain.lower()
        self._model_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )
        self.model = model
        self.scoreboard: Dict[str, Dict[str, TextMessage | int]] = {}
        self.sample_times = sample_times
        self.prune_flag = prune_flag
        self.pass_rate = pass_rate
        self.reflection_records = []
        
        # --- [参数绑定] ---
        self.force_direct_search = force_direct_search
        self.direct_k = direct_k
        self.retrieve_p = retrieve_p
        self.select_q = select_q
        # [新增] 保存
        self.random_k = random_k
        self.random_k_min = int(os.environ.get("MASRUBRIC_RANDOM_K_MIN", "0") or "0")
        self.random_k_max = int(os.environ.get("MASRUBRIC_RANDOM_K_MAX", "0") or "0")
        self.lock_metrics_after_first_round = lock_metrics_after_first_round
        self.use_simple_audit = use_simple_audit
        self.batch_audit_metrics = _read_env_flag("MASRUBRIC_BATCH_AUDIT_METRICS", False)
        self.exact_select_q = _read_env_flag("MASRUBRIC_EXACT_SELECT_Q", False)
        self.batch_audit_max_tokens = int(os.environ.get("MASRUBRIC_BATCH_AUDIT_MAX_TOKENS", "4000"))
        self.cheap_precheck_enabled = _read_env_flag("MASRUBRIC_CHEAP_PRECHECK", False)
        self.cheap_precheck_model = os.environ.get("MASRUBRIC_CHEAP_PRECHECK_MODEL", model)
        self.cheap_precheck_url = os.environ.get("MASRUBRIC_CHEAP_PRECHECK_URL", base_url)
        self.cheap_precheck_api_key = os.environ.get("MASRUBRIC_CHEAP_PRECHECK_API_KEY", api_key)
        # Simple gate reuses the full simple-audit JSON workflow, so it needs the
        # same completion budget as the normal audit prompt instead of a 1-word cap.
        self.cheap_precheck_max_tokens = int(os.environ.get("MASRUBRIC_CHEAP_PRECHECK_MAX_TOKENS", "1500"))
        self.cheap_precheck_trace_path = os.environ.get(CHEAP_PRECHECK_TRACE_ENV, "").strip()
        self._cheap_precheck_client = None
        self.profile_schema = MATCH_PROFILE_SCHEMA
        self.profile_aware_retrieval_enabled = self._is_code_domain() and _read_env_flag(
            "MASRUBRIC_PROFILE_AWARE_RETRIEVAL", True
        )
        self.profile_hard_filter_enabled = self.profile_aware_retrieval_enabled
        # -----------------
        
        if self.use_simple_audit > 0:
            print(f"[Supervisor] Mode: SIMPLE AUDIT (Fixed General Metric). RAG Retrieval is DISABLED.")
        if self.batch_audit_metrics:
            print("[Supervisor] Batch audit enabled. Multiple metrics will be evaluated in one request.")
        if self.exact_select_q:
            print(f"[Supervisor] Exact select_q enabled. Rerank selections will be topped up to {self.select_q} metrics.")
        if self.profile_hard_filter_enabled:
            print("[Supervisor] Profile-aware retrieval enabled with hard filters on interface_shape and io_contract.")
        if self.cheap_precheck_enabled:
            same_endpoint = (
                self.cheap_precheck_model == self.model
                and self.cheap_precheck_url == base_url
                and self.cheap_precheck_api_key == api_key
            )
            if same_endpoint:
                self._cheap_precheck_client = self._model_client
            else:
                self._cheap_precheck_client = AsyncOpenAI(
                    api_key=self.cheap_precheck_api_key,
                    base_url=self.cheap_precheck_url,
                    http_client=DefaultAsyncHttpxClient(trust_env=False),
                )
            print(
                f"[Supervisor] Cheap precheck enabled. "
                f"Model: {self.cheap_precheck_model} | URL: {self.cheap_precheck_url}"
            )
        
        # 1. 初始化 Embedding Client (用于 Query)
        self.embedding_model = embedding_model
        self.embedding_client = OpenAI(
            api_key=embedding_api_key,
            base_url=embedding_api_base,
            http_client=DefaultHttpxClient(trust_env=False),
        )

        # =========================================================
        # [核心优化]：如果传入了预加载数据，直接使用，不再读文件
        # =========================================================
        if preloaded_metrics is not None and preloaded_embeddings is not None:
            # print("[Supervisor] Using preloaded metrics and embeddings (Shared Memory).") # 注释掉以减少日志刷屏
            self.metrics = preloaded_metrics
            self.detailed_definitions_embeddings = preloaded_embeddings
            self._postprocess_metric_pool()
            return # 直接结束初始化
        # =========================================================

        # 下面是旧的“从文件加载”逻辑（保留作为保底）
        if metric_pool_file and os.path.exists(metric_pool_file):
            print(f"[Supervisor] Loading metrics text from: {metric_pool_file}")
            with open(metric_pool_file, "r", encoding='utf-8') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = []
        self._postprocess_metric_pool()

        emb_map = {}
        if embedding_cache_file and os.path.exists(embedding_cache_file):
            print(f"[Supervisor] Loading embedding cache from: {embedding_cache_file}")
            try:
                with open(embedding_cache_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if "name" in record and "vector" in record:
                                emb_map[record["name"]] = record["vector"]
            except Exception as e:
                print(f"[Supervisor Warning] Failed to load embedding cache: {e}")
        
        vectors_list = []
        missing_indices = []
        missing_texts = []
        
        print("[Supervisor] Building vector index...")
        for idx, metric in enumerate(self.metrics):
            name = metric['name']
            if name in emb_map:
                vectors_list.append(np.array(emb_map[name], dtype=np.float32))
            else:
                vectors_list.append(None) 
                missing_indices.append(idx)
                missing_texts.append(build_metric_embedding_text(metric))
        
        if missing_texts:
            print(f"[Supervisor] Calculating {len(missing_texts)} missing embeddings...")
            try:
                response = self.embedding_client.embeddings.create(model=self.embedding_model, input=missing_texts)
                record_embedding_response(
                    response,
                    model=self.embedding_model,
                    stage="embedding_index_build",
                    source="Supervisor",
                    metadata={"missing_metric_count": len(missing_texts)},
                )
                for i, data_item in enumerate(response.data):
                    target_idx = missing_indices[i]
                    vectors_list[target_idx] = np.array(data_item.embedding, dtype=np.float32)
            except Exception as e:
                print(f"[CRITICAL ERROR] Failed to calculate embeddings: {e}")
                for idx in missing_indices:
                    if vectors_list[idx] is None: vectors_list[idx] = np.zeros(1024, dtype=np.float32)

        if vectors_list:
            self.detailed_definitions_embeddings = np.stack(vectors_list)
        else:
            self.detailed_definitions_embeddings = np.array([])
            
        print(f"[Supervisor] Index ready. Shape: {self.detailed_definitions_embeddings.shape}")

    # ==========================================================================
    # [新增] 通用安全解析函数：专治各种 JSON 疑难杂症
    # ==========================================================================
    def _safe_parse_json(self, raw_content: str, source_stage: str) -> Dict:
        """
        尝试从 LLM 输出中提取并解析 JSON。如果失败或类型不对，抛出详细异常。
        """
        try:
            # 1. 尝试正则提取 JSON 块
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                candidate_str = json_match.group(0)
            else:
                # 如果没找到 {}，可能整个就是 JSON，或者整个是乱码
                candidate_str = raw_content

            # 2. 使用 json_repair 强力修复
            # return_objects=True 让它直接返回 Python 对象
            parsed_data = repair_json(candidate_str, return_objects=True)

            # 3. 防御性拆包：有时候会返回 [dict]
            if isinstance(parsed_data, list):
                if len(parsed_data) > 0:
                    parsed_data = parsed_data[0]
                else:
                    raise ValueError("Parsed JSON is an empty list.")

            # 4. 核心类型检查：必须是 Dict
            if not isinstance(parsed_data, dict):
                raise ValueError(f"Parsed data is Type {type(parsed_data)}, NOT dict. Content: {str(parsed_data)[:100]}...")

            return parsed_data

        except Exception as e:
            # 打印详细的错误现场，方便调试
            print(f"\n[JSON Parse Error in {source_stage}]")
            print(f"Error: {e}")
            print(f"Raw Content Snippet: {raw_content[:200]}...") # 只打印前200字符避免刷屏
            raise e # 抛出异常，让上层逻辑（如 fallback）接管
    
    # ==========================================================================

    def _is_code_domain(self) -> bool:
        return any(token in self.domain for token in ("code", "mbpp", "humaneval"))

    def _postprocess_metric_pool(self) -> None:
        if not self.profile_aware_retrieval_enabled:
            return
        for metric in self.metrics:
            metric["match_profile"] = normalize_match_profile(metric.get("match_profile"), metric_side=True)

    def _build_metric_prompt(self, metric: Dict, task: str, role: str, agent_output: str, template: str) -> str:
        m_eval = metric.get("evaluator_prompt", {})
        trigger = m_eval.get("trigger_condition", "N/A")
        risk_alert = m_eval.get("risk_alert", "")
        audit_context = f"Context: {trigger}\nSpecific Risk: {risk_alert}"
        return template.format(
            task=task,
            role=role,
            agent_output=agent_output,
            trigger_condition=audit_context,
        )

    def _build_batch_metric_prompt(self, metrics: List[Dict], task: str, role: str, agent_output: str, template: str) -> str:
        metric_blocks = []
        for index, metric in enumerate(metrics, start=1):
            m_eval = metric.get("evaluator_prompt", {})
            profile = normalize_match_profile(metric.get("match_profile"), metric_side=True)
            metric_blocks.append(
                (
                    f"{index}. Metric Name: {metric['name']}\n"
                    f"   Trigger Condition: {m_eval.get('trigger_condition', 'N/A')}\n"
                    f"   Detailed Definition: {metric.get('detailed_definition', 'N/A')}\n"
                    f"   Specific Risk Alert: {m_eval.get('risk_alert', 'N/A')}"
                    + (
                        f"\n   Interface Shape: {profile['interface_shape']}\n"
                        f"   IO Contract: {profile['io_contract']}\n"
                        f"   Match Hint: {profile['match_hint']}"
                        if self.profile_aware_retrieval_enabled
                        else ""
                    )
                )
            )
        return template.format(
            task=task,
            role=role,
            agent_output=agent_output,
            metrics_block="\n\n".join(metric_blocks),
        )

    async def _build_online_match_profile(self, task: str, agent_output: str) -> Dict[str, str]:
        heuristic_profile = infer_profile_heuristically(task, agent_output)
        if not self._is_code_domain():
            return heuristic_profile

        prompt = MATCH_PROFILE_TEMPLATE.format(task=task, agent_output=agent_output)
        for attempt in range(3):
            completion = None
            raw_content = ""
            try:
                completion = await self._model_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=400,
                )
                record_chat_completion(
                    completion,
                    model=self.model,
                    stage="supervisor_match_profile",
                    source="Supervisor",
                )
                raw_content = completion.choices[0].message.content.strip()
                parsed = self._safe_parse_json(raw_content, source_stage="MatchProfile")
                profile = normalize_match_profile(parsed, metric_side=False)
                if profile.get("match_hint"):
                    return profile
            except Exception as e:
                if attempt == 2:
                    print(f"[Supervisor] Match profile fallback due to: {e}")
        return heuristic_profile

    def _rank_metrics_with_hard_filter(
        self,
        similarities: np.ndarray,
        online_profile: Dict[str, str],
        limit: int,
    ) -> tuple[List[Dict], int]:
        ranked_indices = np.argsort(similarities)[::-1]
        selected: List[Dict] = []
        filtered_count = 0
        for idx in ranked_indices:
            metric = self.metrics[int(idx)]
            if self.profile_hard_filter_enabled and has_profile_conflict(metric.get("match_profile"), online_profile):
                filtered_count += 1
                continue
            selected.append(metric)
            if len(selected) >= limit:
                break
        return selected, filtered_count

    def _finding_to_judgement(self, metric: Dict, finding: Dict) -> Dict:
        evidence = finding.get("evidence_quote", "N/A")
        analysis = finding.get("analysis", "N/A")
        suggestion = finding.get("suggestion", "N/A")
        impact = finding.get("impact_assessment", "NO")
        raw_flawed = finding.get("is_flawed", False)

        is_suggestion_valid = str(suggestion).lower() not in ["n/a", "none", "no suggestion", "", "null"]
        is_impact_significant = "yes" in str(impact).lower()

        if raw_flawed:
            if not is_suggestion_valid:
                final_verdict_bool = False
            elif not is_impact_significant:
                final_verdict_bool = False
            else:
                final_verdict_bool = True
        else:
            final_verdict_bool = False

        verdict_str = "flawed" if final_verdict_bool else "correct"
        return {
            "metric": metric["name"],
            "verdict": verdict_str,
            "evidence_quote": evidence,
            "reasoning": analysis,
            "suggestion": suggestion,
            "impact": impact,
            "is_triggered": True,
        }

    def _append_cheap_precheck_trace(self, payload: Dict) -> None:
        if not self.cheap_precheck_trace_path:
            return
        trace_path = Path(self.cheap_precheck_trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        event = {"timestamp": _utc_now_iso(), **payload}
        with _CHEAP_PRECHECK_TRACE_LOCK:
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False))
                f.write("\n")

    def _extract_batch_judgements(self, payload: Dict[str, Any], metrics: List[Dict]) -> List[Dict]:
        raw_items: Any = payload.get("judgements")
        if raw_items is None:
            raw_items = payload.get("audits")
        if raw_items is None:
            raw_items = payload.get("results")
        if raw_items is None:
            metrics_payload = payload.get("metrics")
            if isinstance(metrics_payload, dict):
                raw_items = []
                for name, item in metrics_payload.items():
                    if isinstance(item, dict):
                        raw_items.append({"metric": name, **item})
        if not isinstance(raw_items, list):
            raise ValueError("Batch audit response does not contain a valid judgement list.")

        metric_map = {metric["name"]: metric for metric in metrics}
        raw_by_name: dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Batch audit item at index {index} is not a JSON object.")
            metric_name = item.get("metric") or item.get("metric_name") or item.get("name")
            if not metric_name and index < len(metrics):
                metric_name = metrics[index]["name"]
            if metric_name not in metric_map:
                raise ValueError(f"Unknown metric returned by batch audit: {metric_name}")
            if metric_name in raw_by_name:
                raise ValueError(f"Duplicate metric returned by batch audit: {metric_name}")
            raw_by_name[metric_name] = item

        missing = [metric["name"] for metric in metrics if metric["name"] not in raw_by_name]
        if missing:
            raise ValueError(f"Batch audit response is missing metrics: {missing}")

        judgements = []
        for metric in metrics:
            judgements.append(self._finding_to_judgement(metric, raw_by_name[metric["name"]]))
        return judgements

    def _apply_select_q_policy(self, candidates: List[Dict], selected_names: Any) -> List[Dict]:
        if not isinstance(selected_names, list):
            selected_names = []

        selected_name_set = {str(name) for name in selected_names}
        matched_metrics = [m for m in candidates if m.get('name') in selected_name_set]

        if self.exact_select_q:
            matched_names = {m.get('name') for m in matched_metrics}
            for candidate in candidates:
                if len(matched_metrics) >= self.select_q:
                    break
                candidate_name = candidate.get('name')
                if candidate_name in matched_names:
                    continue
                matched_metrics.append(candidate)
                matched_names.add(candidate_name)

        return matched_metrics[:self.select_q]

    async def _match_metrics(self, task: str, output: str, metric_names=None) -> List[Dict]:
        # 1. 如果指定了具体名字（复用逻辑），直接返回
        if metric_names is not None:
            return [m for m in self.metrics if m['name'] in metric_names]

        # =======================================================
        # [新增] 随机对照组逻辑：只要 k > 0，直接随机抽，无视 RAG
        # =======================================================
        if self.random_k > 0:
            # 防止 k 大于池子总数报错
            if self.random_k_min > 0 and self.random_k_max >= self.random_k_min:
                sampled_k = random.randint(self.random_k_min, self.random_k_max)
                print(
                    f"[Supervisor] Mode: Random Selection "
                    f"(k=random {self.random_k_min}-{self.random_k_max}, sampled={sampled_k})"
                )
            else:
                sampled_k = self.random_k
                print(f"[Supervisor] Mode: Random Selection (k={sampled_k})")
            k = min(sampled_k, len(self.metrics))
            return random.sample(self.metrics, k)
        # =======================================================

        if self.profile_aware_retrieval_enabled:
            online_profile = await self._build_online_match_profile(task, output)
            query_text = build_query_embedding_text(online_profile)
            print(f"Generated Match Profile: {json.dumps(online_profile, ensure_ascii=False)}")
            embedding_stage = "embedding_query_profileaware"
        else:
            online_profile = {}
            summary_resp = await self._model_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": SUMMARY_TEMPLATE.format(task=task, agent_output=output)}],
                temperature=0.0,
                max_tokens=1000,
            )
            record_chat_completion(summary_resp, model=self.model, stage="supervisor_summary", source="Supervisor")
            raw_summary = summary_resp.choices[0].message.content.strip()
            try:
                json_match = re.search(r"\{.*\}", raw_summary, re.DOTALL)
                summary_data = json.loads(json_match.group(0)) if json_match else {}
                query_text = (
                    "Problem Scenario: " + ", ".join(summary_data.get("problem_scenario", []))
                    + ". Agent Action: " + ", ".join(summary_data.get("agent_action", []))
                )
            except Exception:
                query_text = raw_summary
            embedding_stage = "embedding_query"
        print(f"Generated Search Query: {query_text}")

        # --- Step 2: 向量计算 ---
        emb_resp = self.embedding_client.embeddings.create(model=self.embedding_model, input=[query_text])
        record_embedding_response(
            emb_resp,
            model=self.embedding_model,
            stage=embedding_stage,
            source="Supervisor",
        )
        query_emb = np.array(emb_resp.data[0].embedding, dtype=np.float32)
        similarities = np.dot(self.detailed_definitions_embeddings, query_emb)
        
        
        # =======================================================
        # [核心分支逻辑] Direct Search vs LLM Rerank
        # =======================================================
        
        if self.force_direct_search:
            # --- 模式 A: 强制直选 ---
            print(f"[Supervisor] Mode: Direct Search (Top-{self.direct_k}).")
            selected, filtered_count = self._rank_metrics_with_hard_filter(
                similarities,
                online_profile,
                self.direct_k,
            )
            if self.profile_hard_filter_enabled:
                print(f"[Supervisor] Hard-filtered conflicting metrics: {filtered_count}")
            return selected
            
        else:
            # --- 模式 B: LLM Rerank ---
            select_policy = "Exact Top" if self.exact_select_q else "Up to Top"
            print(f"[Supervisor] Mode: LLM Rerank (Pool Top-{self.retrieve_p} -> Select {select_policy}-{self.select_q}).")
            
            # 1. 粗排召回 (Candidate Pool)
            candidates, filtered_count = self._rank_metrics_with_hard_filter(
                similarities,
                online_profile,
                self.retrieve_p,
            )
            if self.profile_hard_filter_enabled:
                print(f"[Supervisor] Hard-filtered conflicting metrics before rerank: {filtered_count}")

            # 2. 构建 LLM 选择 Prompt
            candidate_str = ""
            for m in candidates:
                candidate_str += f"- Metric: {m['name']}\n  Trigger: {m.get('evaluator_prompt',{}).get('trigger_condition','')}\n  Definition: {m['detailed_definition']}"
                if self.profile_aware_retrieval_enabled:
                    profile = normalize_match_profile(m.get("match_profile"), metric_side=True)
                    candidate_str += (
                        f"\n  Shape: {profile['interface_shape']}"
                        f"\n  IO: {profile['io_contract']}"
                        f"\n  Hint: {profile['match_hint']}"
                    )
                candidate_str += "\n\n"

            selection_prompt = SELECTION_TEMPLATE.format(
                task=task,
                agent_output=output,
                metrics_list=candidate_str,
                selection_count_instruction=(f"exactly {self.select_q}" if self.exact_select_q else f"1-{self.select_q}"),
                select_q=self.select_q,
            )

            try:
                # 3. LLM 精选
                response = await self._model_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": selection_prompt}],
                    temperature=0.0,
                    max_tokens=1500,
                )
                record_chat_completion(response, model=self.model, stage="supervisor_rerank", source="Supervisor")
                raw_content = response.choices[0].message.content.strip()
                
                # 4. 解析结果
                selection_data = self._safe_parse_json(raw_content, source_stage="Rerank")
                selected_names = selection_data.get("selected_metrics", [])

                # 5. 数量策略：默认保留旧的 1-Q 流动选择；固定 topN 模式下补齐到 Q。
                return self._apply_select_q_policy(candidates, selected_names)
                
            except Exception as e:
                print(f"[Rerank Fallback] Using Direct Top-{self.select_q} due to: {e}")
                # 保底：如果 LLM 挂了，直接从候选池里取前 Q 个
                return candidates[:self.select_q]
   
   
    
    def _parse_score(self, response: str) -> int:
        
        # 从末尾开始匹配 <Score> 标签后的数字
        match = re.search(r'<Score>\s*(\d+)\s*$', response.strip(), re.MULTILINE)
        if match:
            return int(match.group(1))
        
        match_2 = re.search(r'</Score>\s*(\d+)\s*$', response.strip(), re.MULTILINE)
        if match_2:
            return int(match_2.group(1))
        raise ValueError(f"No valid score found in response: {response}")

    def _parse_cheap_precheck_verdict(self, response: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(response or "")).strip().upper()
        if not normalized:
            return True

        token_match = re.search(r"\b(CHECK|PASS)\b", normalized)
        if token_match:
            return token_match.group(1) == "CHECK"

        if any(flag in normalized for flag in ["YES", "RISK", "ISSUE", "PROBLEM", "ERROR"]):
            return True
        if any(flag in normalized for flag in ["NO", "SAFE", "OK", "PASS"]):
            return False
        return True

    async def _run_cheap_precheck(self, task: str, agent_output: str, role: str, *, message_id: str | None = None, message_source: str | None = None) -> tuple[bool, str]:
        if not self.cheap_precheck_enabled or self._cheap_precheck_client is None:
            return True, "CHECK"

        normalized_role = str(role or "").strip().upper()
        if normalized_role in CHEAP_PRECHECK_SKIP_ROLES:
            raw_content = "SKIP_PROGRAMMING_EXPERT"
            self._append_cheap_precheck_trace(
                {
                    "stage": "supervisor_cheap_precheck",
                    "gate_mode": "simple_audit_v2",
                    "task": task,
                    "role": role,
                    "message_id": message_id,
                    "message_source": message_source,
                    "metric": None,
                    "model": self.cheap_precheck_model,
                    "base_url": self.cheap_precheck_url,
                    "raw_response": raw_content,
                    "judgement": None,
                    "finding": None,
                    "should_audit": False,
                    "passed_gate": True,
                    "skipped": True,
                }
            )
            return False, raw_content

        metric = SIMPLE_CODE_METRIC_2 if self._is_code_domain() else SIMPLE_MATH_METRIC_2
        target_template = METRIC_TEMPLATE_CODE_AUDIT if self._is_code_domain() else METRIC_TEMPLATE_SINGLE_AUDIT
        prompt = self._build_metric_prompt(metric, task, role, agent_output, target_template)

        for attempt in range(5):
            completion = None
            raw_content = ""
            try:
                completion = await self._cheap_precheck_client.chat.completions.create(
                    model=self.cheap_precheck_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=self.cheap_precheck_max_tokens,
                )
                raw_content = completion.choices[0].message.content.strip()
                finding = self._safe_parse_json(raw_content, source_stage=f"CheapPrecheck-{metric['name']}")
                judgement = self._finding_to_judgement(metric, finding)
                should_audit = judgement["verdict"].lower() != "correct"

                record_chat_completion(
                    completion,
                    model=self.cheap_precheck_model,
                    stage="supervisor_cheap_precheck",
                    source="Supervisor",
                    metadata={
                        "gate_mode": "simple_audit_v2",
                        "role": role,
                        "metric": metric["name"],
                        "verdict": judgement["verdict"],
                        "message_id": message_id,
                        "message_source": message_source,
                        "should_audit": should_audit,
                    },
                )
                self._append_cheap_precheck_trace(
                    {
                        "stage": "supervisor_cheap_precheck",
                        "gate_mode": "simple_audit_v2",
                        "task": task,
                        "role": role,
                        "message_id": message_id,
                        "message_source": message_source,
                        "metric": metric["name"],
                        "model": self.cheap_precheck_model,
                        "base_url": self.cheap_precheck_url,
                        "raw_response": raw_content,
                        "judgement": judgement,
                        "finding": finding,
                        "should_audit": should_audit,
                        "passed_gate": not should_audit,
                        "skipped": False,
                    }
                )
                return should_audit, raw_content
            except Exception as e:
                if completion is not None:
                    record_chat_completion(
                        completion,
                        model=self.cheap_precheck_model,
                        stage="supervisor_cheap_precheck",
                        source="Supervisor",
                        metadata={
                            "gate_mode": "simple_audit_v2",
                            "role": role,
                            "metric": metric["name"],
                            "message_id": message_id,
                            "message_source": message_source,
                            "parse_error": str(e),
                        },
                    )
                if attempt == 4:
                    self._append_cheap_precheck_trace(
                        {
                            "stage": "supervisor_cheap_precheck",
                            "gate_mode": "simple_audit_v2",
                            "task": task,
                            "role": role,
                            "message_id": message_id,
                            "message_source": message_source,
                            "metric": metric["name"],
                            "model": self.cheap_precheck_model,
                            "base_url": self.cheap_precheck_url,
                            "raw_response": raw_content,
                            "judgement": None,
                            "finding": None,
                            "should_audit": True,
                            "passed_gate": False,
                            "skipped": False,
                            "error": str(e),
                        }
                    )
                    print(f"[Cheap Precheck] Simple-audit gate failed after 5 attempts: {e}")
                    return True, raw_content or "CHECK"
    
# 需要增加 role 参数，或者从 message 对象中推断 role
    # [修改] _calc_score 方法签名，接收 metrics 列表
    async def _calc_score_individual(
        self,
        task,
        message_content: str,
        role: str,
        matched_metrics: List[Dict],
        target_template: str,
        stage_name: str,
    ) -> List[Dict]:
        judgements = []
        for metric in matched_metrics:
            prompt = self._build_metric_prompt(metric, task, role, message_content, target_template)

            for attempt in range(5):
                try:
                    completion = await self._model_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1500
                    )
                    record_chat_completion(
                        completion,
                        model=self.model,
                        stage=stage_name,
                        source="Supervisor",
                    )
                    res_raw = completion.choices[0].message.content.strip()

                    finding = self._safe_parse_json(res_raw, source_stage=f"Audit-{metric['name']}")
                    judgement = self._finding_to_judgement(metric, finding)
                    judgements.append(judgement)
                    break

                except Exception as e:
                    if attempt == 4:
                        print(f"[Audit Fail] Metric '{metric['name']}' failed 5 times. Last Error: {e}")

        return judgements

    async def _calc_score_batch(
        self,
        task,
        message_content: str,
        role: str,
        matched_metrics: List[Dict],
        target_template: str,
        stage_name: str,
    ) -> List[Dict]:
        prompt = self._build_batch_metric_prompt(
            matched_metrics,
            task=task,
            role=role,
            agent_output=message_content,
            template=target_template,
        )
        for attempt in range(5):
            try:
                completion = await self._model_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=self.batch_audit_max_tokens,
                )
                record_chat_completion(
                    completion,
                    model=self.model,
                    stage=f"{stage_name}_batch",
                    source="Supervisor",
                    metadata={"metric_count": len(matched_metrics)},
                )
                res_raw = completion.choices[0].message.content.strip()
                payload = self._safe_parse_json(res_raw, source_stage="AuditBatch")
                return self._extract_batch_judgements(payload, matched_metrics)
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"[Audit Batch Retry] attempt={attempt + 1} error={e}")

        return []

    async def _calc_score(self, task, message: TextMessage | dict, role: str = "Unknown", metrics=None) -> List[Dict]:
        message_content = message.content if isinstance(message, TextMessage) else message['content']
        
        # 1. 确定使用的指标
        if metrics is not None:
            matched_metrics = metrics
        else:
            matched_metrics = await self._match_metrics(task, message_content)
        
        # -------------------------------------------------------------
        # [核心修改] 模板选择逻辑
        # -------------------------------------------------------------
        if self.random_k > 0:
            # === Random Ablation 模式 (使用官僚模板) ===
            # print(f"[Audit] Using STRICT RANDOM TEMPLATE for ablation.") 
            if self.domain == "code":
                target_template = METRIC_TEMPLATE_RANDOM_CODE
            else:
                target_template = METRIC_TEMPLATE_RANDOM_MATH
        elif self.domain == "code":
            # === 正常代码模式 ===
            target_template = METRIC_TEMPLATE_CODE_AUDIT
        else:
            # === 正常数学模式 ===
            target_template = METRIC_TEMPLATE_SINGLE_AUDIT
        # -------------------------------------------------------------
        stage_name = "supervisor_simple_audit" if self.use_simple_audit > 0 else "supervisor_audit"
        if not matched_metrics:
            return []

        if self.batch_audit_metrics and len(matched_metrics) > 1:
            batch_template = METRIC_TEMPLATE_BATCH_AUDIT_CODE if self._is_code_domain() else METRIC_TEMPLATE_BATCH_AUDIT_MATH
            try:
                return await self._calc_score_batch(
                    task=task,
                    message_content=message_content,
                    role=role,
                    matched_metrics=matched_metrics,
                    target_template=batch_template,
                    stage_name=stage_name,
                )
            except Exception as e:
                print(f"[Audit Batch Fallback] Falling back to per-metric audit due to: {e}")

        return await self._calc_score_individual(
            task=task,
            message_content=message_content,
            role=role,
            matched_metrics=matched_metrics,
            target_template=target_template,
            stage_name=stage_name,
        )
    
    
    
    
    # 1. 常规更新 (通常用于流式或单步调试，建议同步修改以防万一)
    async def update_scoreboard(self, task: str, message: TextMessage):
        if not self.prune_flag:
            self.scoreboard[message.id] = {
                "message": message,
                "judgements": [],
                "is_pruned": False # <--- 强制 False
            }
        else:
            judgements = await self._calc_score(task, message)
            pass_cnt = 0
            for judge in judgements:
                if judge['verdict'].lower() == 'correct':
                    pass_cnt += 1
            
            # 这里的 else False 很关键
            is_pruned = (pass_cnt / len(judgements)) < self.pass_rate if judgements else False
            
            self.scoreboard[message.id] = {
                "message": message,
                "judgements": judgements,
                "is_pruned": is_pruned
            }

    # 2. 结果回填更新 (这是 MathSolver_aqua 主要调用的方法，必须改！)
    def update_scoreboard_with_results(self, message: TextMessage, judgements: list[dict]):
        # [新增] 防御性编程：如果全局没开剪枝，直接通过
        if not self.prune_flag:
            self.scoreboard[message.id] = {
                "message": message,
                "judgements": judgements, # 此时通常是 []
                "is_pruned": False        # <--- 核心修正
            }
            return

        pass_cnt = 0
        for judge in judgements:
            if judge['verdict'].lower() == 'correct':
                pass_cnt += 1
        
        # [核心修正] 如果没有指标覆盖(judgements为空)，默认为 False (不剪枝/无罪释放)
        is_pruned = (pass_cnt / len(judgements)) < self.pass_rate if judgements else False

        self.scoreboard[message.id] = {
            "message": message,
            "judgements": judgements,
            "is_pruned": is_pruned
        }
        
# 此处逻辑基本保持，但需确保判定 correct 的逻辑稳健
    # [修改] judge 方法签名，增加 session_metrics 用于回传
    async def judge(self, task: str, message: TextMessage, attempt_num: int, role: str = "Assistant", previous_metrics=None, session_metrics=None):
        # 如果非剪枝模式，直接跳过
        if not self.prune_flag:
            return True, [], None, None 

        message_content = message.content if isinstance(message, TextMessage) else message['content']

        print("\n" + "="*80)
        print(f"--- 审计轮次: {attempt_num} | Agent: {message.source} (Role: {role}) ---")
        print("="*80)

        if self.cheap_precheck_enabled:
            try:
                message_id = message.id if isinstance(message, TextMessage) else message.get("id")
                message_source = message.source if isinstance(message, TextMessage) else message.get("source")
                should_audit, precheck_raw = await self._run_cheap_precheck(
                    task,
                    message_content,
                    role,
                    message_id=message_id,
                    message_source=message_source,
                )
                print(f"[Cheap Precheck] Raw verdict: {precheck_raw}")
                if not should_audit:
                    print("[Cheap Precheck] PASS -> Skip full audit and let the message pass.")
                    print("="*80 + "\n")
                    return True, [], None, []
                print("[Cheap Precheck] CHECK -> Continue to the main V2 audit pipeline.")
            except Exception as e:
                print(f"[Cheap Precheck] Failed, fallback to main V2 audit pipeline: {e}")
        
        current_metrics = []

        # ------------------------------------------------------------------
        # [核心修改] 分支逻辑：简单模式 (V1/V2) vs RAG 模式
        # ------------------------------------------------------------------
        if self.use_simple_audit > 0:
            
            # === V1: 通用简单审计 (旧版) ===
            if self.use_simple_audit == 1:
                print("[Supervisor] Using FIXED General Metric (Simple Audit V1).")
                # 根据 domain 选择 V1 指标
                if "code" in self.domain or "mbpp" in self.domain:
                    current_metrics = [SIMPLE_CODE_METRIC_1] # 需确保你在文件头定义了这个变量
                else:
                    current_metrics = [SIMPLE_MATH_METRIC_1]

            # === V2: 优化版简单审计 (新版) ===
            elif self.use_simple_audit == 2:
                print("[Supervisor] Using OPTIMIZED General Metric (Simple Audit V2).")
                # 根据 domain 选择 V2 指标
                if "code" in self.domain or "mbpp" in self.domain:
                    current_metrics = [SIMPLE_CODE_METRIC_2] # 需确保你在文件头定义了这个变量
                else:
                    current_metrics = [SIMPLE_MATH_METRIC_2]
            
        else:
            # 模式 B: RAG Audit (原有逻辑)
            # [指标复用逻辑]
            if self.lock_metrics_after_first_round and attempt_num > 1 and session_metrics:
                print("[Supervisor] Using LOCKED metrics from Attempt 1.")
                current_metrics = session_metrics
            else:
                print("[Supervisor] Retrieving NEW metrics from Pool...")
                current_metrics = await self._match_metrics(task, message_content)

        # ------------------------------------------------------------------

        # 计算分数
        judgements = await self._calc_score(task, message, role=role, metrics=current_metrics)
        
        print("\n[1] 选择的审计指标:")
        print(" -> " + ", ".join([j['metric'] for j in judgements]) if judgements else " (无)")

        print("\n[2] 各项审计结果:")
        pass_cnt = 0
        feedback_lines = []
        
        for j in judgements:
            is_correct = (j['verdict'].lower() == 'correct')
            metric_name = j['metric']
            
            if is_correct: 
                pass_cnt += 1
                status = "✅ Correct"
                print(f" - {metric_name}: {status}")
            else:
                status = "❌ Flawed"
                print(f" - {metric_name}: {status}")
                
                # --- [详细日志打印] ---
                evidence = j.get('evidence_quote', 'N/A')
                reason = j.get('reasoning', 'N/A')
                suggestion = j.get('suggestion', 'N/A')
                impact = j.get('impact', 'N/A')
                
                print(f"   - 证据: {evidence}")
                print(f"   - 分析: {reason}")
                print(f"   - 建议: {suggestion}")
                print(f"   - 影响: {impact}")
                
                # --- [组装反馈] ---
                # 只有当有明确建议时，反馈才有价值
                # [修改 1] 极简压缩格式
                # 格式：- [指标名]: 建议 (原因摘要)
                # 截断 reason 防止太长
                #short_reason = (reason[:200] + '...') if len(reason) > 200 else reason
                #改得稍微大一点，防止不完全
                short_reason = (reason[:1000] + '...') if len(reason) > 1000 else reason
                
                item = (
                    f"- [{metric_name}]: {suggestion}\n"
                    f"  (Auditor's Note: {short_reason})"
                )
                feedback_lines.append(item)
        
        # 3. 计算通过状态
        total_metrics = len(judgements)
        pass_flag = (pass_cnt / total_metrics) >= self.pass_rate if total_metrics > 0 else True 
        
        # 4. 生成最终 Feedback 字符串
        # 只有在判定为 False 时才生成反馈，避免干扰
        if not pass_flag and feedback_lines:
            feedback_body = "\n".join(feedback_lines)
            feedback = (
                f"An external auditor has reviewed your previous output (Attempt {attempt_num}) and flagged some potential issues. "
                "Please review the following suggestions critically:\n\n"
                f"{feedback_body}\n\n"
                "**Instruction**:\n"
                "1. If you agree with the advice, please refine your solution.\n"
                "2. **If you are confident your original logic is correct, you may ignore this advice.**\n"
                "3. Please output the corrected solution."
            )
        else:
            feedback = None
            
        print("\n[3] 综合判定:")
        print(f" -> 通过率: {pass_cnt}/{total_metrics} | 阈值: {self.pass_rate:.0%} | 结果: {'通过' if pass_flag else '失败'}")
        
        if feedback:
            print("\n[4] 发送给 Agent 的反馈:")
            print(feedback)
            
        print("="*80 + "\n")
        
        return pass_flag, judgements, feedback, current_metrics
    
    
    
        
    def prune_info(self, all_messages: List[TextMessage]) -> List[TextMessage]:
        # if self.threshold == 0.0:
        if not self.prune_flag:
            return all_messages
        pruned_messages = []
        for msg in all_messages:
            if msg.id in self.scoreboard:
                # if self.scoreboard[msg.id]["avg"] >= self.threshold:
                #     pruned_messages.append(msg)
                # else:
                #     print(f"Info: Message ID {msg.id} filtered out by supervisor due to low score.")
                # if "flawed" not in [judge['verdict'] for judge in self.scoreboard[msg.id]['judgements']]:
                # if pass_cnt >= self.metrics_pass_k:
                if not self.scoreboard[msg.id]['is_pruned']:
                    pruned_messages.append(msg)
                else:
                    print(f"Info: Message ID {msg.id} filtered out by supervisor due to flawed judgement.")
            else:
                if msg.source != "user":
                    print(f"Warning: Message ID {msg.id} not found in scoreboard. Passing by default...")
                pruned_messages.append(msg)
        return pruned_messages
    
    def get_messages_above_threshold(self) -> List[TextMessage]:
        # if self.threshold == 0.0:
        if not self.prune_flag:
            return [entry["message"] for entry in self.scoreboard.values()]
        # return [entry["message"] for entry in self.scoreboard.values() if entry["avg"] >= self.threshold]
        return [entry["message"] for entry in self.scoreboard.values() if not entry["is_pruned"]]
    
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
                # 'scores': {**{metric: self.scoreboard[entry][metric] for metric in self.metrics}, 'avg': self.scoreboard[entry]['avg']},
                'judgements': self.scoreboard[entry]['judgements'],
                'is_pruned': self.scoreboard[entry]['is_pruned'],
            }
        return ret_scores
    
    def reset(self):
        self.scoreboard = {}
