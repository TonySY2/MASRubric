from masrubric.usage import tracked_create, tracked_embedding_create
#--- START OF FILE supervisor_reasoning_pick_metric.py ---

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from typing import Any, Dict, List
from autogen_core.models import UserMessage, ModelInfo
import re
from openai import AsyncOpenAI
import json
from openai import OpenAI
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from json_repair import repair_json
import os
import random 







SIMPLE_MATH_METRIC_1 = {
    "name": "CRITICAL_MATH_LOGIC_AUDIT",
    "detailed_definition": "A focused audit to detect substantive logical fallacies, calculation errors, or conditional oversights that invalidate the final result.",
    "evaluator_prompt": {
    
        "trigger_condition": "The Agent is performing mathematical reasoning, derivation, or calculation.",
        
   
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

SIMPLE_CODE_METRIC_1 = {
    "name": "CRITICAL_CODE_CORRECTNESS_CHECK",
    "detailed_definition": "A functional audit focusing on runtime safety, logical integrity, and adherence to requirements in code implementation.",
    "evaluator_prompt": {
      
        "trigger_condition": "The Agent is generating, debugging, or analyzing computer code.",
        
    
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


METRIC_TEMPLATE_MATH_AUDIT = """
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

RERANK_TEMPLATE = """[Instruction]
Select the most relevant indicators for auditing the current agent output.
You receive candidates from embedding retrieval. Return only indicator names from the candidate list.

Selection rule:
- Select at most {select_q} indicators.
- If exact_selection is true, select exactly {select_q} indicators unless fewer candidates are available.
- Prefer indicators that can expose concrete fatal reasoning or coding errors in the agent output.

Context:
Task: {task}
Agent output: {agent_output}
exact_selection: {exact_selection}

Candidates:
{candidates}

JSON output only:
{{
  "selected_names": ["indicator_name"]
}}
"""

BATCH_METRIC_TEMPLATE = """
You are auditing one agent output against multiple indicators in a single pass.
For each indicator, decide whether it exposes a fatal flaw in the output.

Audit protocol:
1. Assume the output is valid unless there is concrete evidence of a fatal flaw.
2. If an indicator is irrelevant to this task/output, mark it as not flawed.
3. A flaw must be actionable and must change the final answer, runtime behavior, or core conclusion.

Context:
- Domain: {domain}
- Task: {task}
- Agent Role: {role}
- Agent Output: {agent_output}

Indicators:
{indicator_block}

Return JSON only as an array. Use one object per indicator:
[
  {{
    "metric": "indicator_name",
    "evidence_quote": "verbatim quote, or N/A",
    "analysis": "concise reason, or N/A",
    "suggestion": "specific correction, or N/A",
    "impact_assessment": "YES/NO and brief reason",
    "is_flawed": false
  }}
]
"""

class Supervisor():
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        domain: str,  
        
 
        direct_k: int = 5,
        random_k: int = 0,
        random_k_min: int = 0,
        random_k_max: int = 0,
        retrieval_mode: str = "direct",
        retrieve_p: int = 20,
        select_q: int | None = None,
        exact_select_q: bool = False,
        batch_audit_metrics: bool = False,
        metrics_retrieve_k: int | None = None,
        use_llm_rerank: bool | None = None,
        max_metrics_count: int | None = None,
        force_direct_search: bool = False,
        use_simple_audit: int = 0,         
        
        sample_times: int = 3,
        pass_rate: float = 1,
        prune_flag: bool = True,
        
   
        metric_pool_file: str = "",
        embedding_cache_file: str = "",
        embedding_model: str = "",
        embedding_api_key: str = "",
        embedding_api_base: str = "",
        

        preloaded_metrics: List[Dict] = None,
        preloaded_embeddings: np.ndarray = None,
        **_ignored_kwargs: Any,
    ):
        self.domain = domain.lower()
        self._model_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.scoreboard: Dict[str, Dict] = {}
        self.sample_times = sample_times
        self.prune_flag = prune_flag
        self.pass_rate = pass_rate
        self.reflection_records = []
        

        self.direct_k = max(1, direct_k)
        self.random_k = max(0, random_k)
        self.random_k_min = max(0, random_k_min)
        self.random_k_max = max(0, random_k_max)
        if metrics_retrieve_k is not None:
            retrieve_p = metrics_retrieve_k
        if max_metrics_count is not None and select_q is None:
            select_q = max_metrics_count
        if use_llm_rerank is True and not force_direct_search:
            retrieval_mode = "rerank"
        if force_direct_search:
            retrieval_mode = "direct"
        self.retrieval_mode = (retrieval_mode or "direct").lower()
        if self.random_k > 0 or self.random_k_max > 0:
            self.retrieval_mode = "random"
        if self.retrieval_mode not in {"direct", "rerank", "random"}:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
        self.retrieve_p = max(1, retrieve_p)
        self.select_q = max(1, select_q if select_q is not None else self.direct_k)
        self.exact_select_q = bool(exact_select_q)
        self.batch_audit_metrics = bool(batch_audit_metrics)
        self.use_simple_audit = use_simple_audit
        
        if self.use_simple_audit > 0:
            print(f"[Supervisor] Mode: SIMPLE AUDIT (Fixed General Metric). RAG Retrieval is DISABLED.")
        
        # Embedding Client
        self.embedding_model = embedding_model
        self.embedding_client = OpenAI(
            api_key=embedding_api_key,
            base_url=embedding_api_base,
        )

        if preloaded_metrics is not None and preloaded_embeddings is not None:
            self.metrics = preloaded_metrics
            self.detailed_definitions_embeddings = preloaded_embeddings
            return 

  
        if metric_pool_file and os.path.exists(metric_pool_file):
            print(f"[Supervisor] Loading metrics text from: {metric_pool_file}")
            with open(metric_pool_file, "r", encoding='utf-8') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = []

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
        missing_texts = []
        missing_indices = []
        
        print("[Supervisor] Building vector index...")
        for idx, metric in enumerate(self.metrics):
            name = metric['name']
            if name in emb_map:
                vectors_list.append(np.array(emb_map[name], dtype=np.float32))
            else:
                vectors_list.append(None) 
                missing_indices.append(idx)
                trigger_text = metric.get('evaluator_prompt', {}).get('trigger_condition', metric['detailed_definition'])
                missing_texts.append(trigger_text)
        
        if missing_texts:
            print(f"[Supervisor] Calculating {len(missing_texts)} missing embeddings...")
            try:
                response = tracked_embedding_create(self.embedding_client.embeddings.create, stage="embedding_pool_setup", source="Supervisor", model=self.embedding_model, input=missing_texts)
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

    def _safe_parse_json_any(self, raw_content: str, source_stage: str) -> Any:
        try:
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            candidate_str = json_match.group(0) if json_match else raw_content
            if "[" in raw_content and "]" in raw_content:
                list_match = re.search(r'\[.*\]', raw_content, re.DOTALL)
                if list_match and (not json_match or list_match.start() <= json_match.start()):
                    candidate_str = list_match.group(0)
            parsed_data = repair_json(candidate_str, return_objects=True)
            return parsed_data
        except Exception as e:
            print(f"\n[JSON Parse Error in {source_stage}]: {e}")
            raise e

    def _safe_parse_json(self, raw_content: str, source_stage: str) -> Dict:

        try:
            parsed_data = self._safe_parse_json_any(raw_content, source_stage)
            if isinstance(parsed_data, list):
                if len(parsed_data) > 0: parsed_data = parsed_data[0]
                else: raise ValueError("Parsed JSON is an empty list.")
            if not isinstance(parsed_data, dict):
                raise ValueError(f"Parsed data is Type {type(parsed_data)}, NOT dict.")
            return parsed_data
        except Exception as e:
            print(f"\n[JSON Parse Error in {source_stage}]: {e}")
            raise e

    def _is_truthy(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "y", "1", "flawed"}
        return bool(value)

    def _judgement_from_finding(self, metric_name: str, finding: Dict[str, Any]) -> Dict[str, Any]:
        evidence = finding.get("evidence_quote", "N/A")
        analysis = finding.get("analysis", "N/A")
        suggestion = finding.get("suggestion", "N/A")
        impact = finding.get("impact_assessment", "NO")
        raw_flawed = self._is_truthy(finding.get("is_flawed", False))

        is_suggestion_valid = str(suggestion).lower() not in ["n/a", "none", "no suggestion", "", "null"]
        is_impact_significant = "yes" in str(impact).lower()
        final_verdict_bool = raw_flawed and is_suggestion_valid and is_impact_significant
        verdict_str = "flawed" if final_verdict_bool else "correct"

        return {
            "metric": metric_name,
            "verdict": verdict_str,
            "evidence_quote": evidence,
            "reasoning": analysis,
            "suggestion": suggestion,
            "impact": impact,
            "is_triggered": True,
        }

    def _format_metric_candidates(self, metrics: List[Dict], include_risk: bool = True) -> str:
        lines = []
        for idx, metric in enumerate(metrics, start=1):
            evaluator_prompt = metric.get("evaluator_prompt", {})
            trigger = evaluator_prompt.get("trigger_condition", "N/A")
            risk = evaluator_prompt.get("risk_alert", "")
            definition = metric.get("detailed_definition", "")
            item = [
                f"{idx}. name: {metric.get('name', '')}",
                f"   definition: {definition}",
                f"   trigger: {trigger}",
            ]
            if include_risk and risk:
                item.append(f"   risk: {risk}")
            lines.append("\n".join(item))
        return "\n\n".join(lines)

    def _top_metric_candidates(self, query_emb: np.ndarray, k: int) -> List[Dict]:
        if self.detailed_definitions_embeddings.size == 0:
            return []
        similarities = np.dot(self.detailed_definitions_embeddings, query_emb)
        top_k_indices = np.argsort(similarities)[-min(k, len(self.metrics)):][::-1]
        return [self.metrics[idx] for idx in top_k_indices]

    async def _rerank_metrics(self, task: str, output: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        fallback = candidates[: min(self.select_q, len(candidates))]
        prompt = RERANK_TEMPLATE.format(
            task=task,
            agent_output=output,
            select_q=self.select_q,
            exact_selection=str(self.exact_select_q).lower(),
            candidates=self._format_metric_candidates(candidates, include_risk=False),
        )
        try:
            completion = await tracked_create(
                self._model_client.chat.completions.create, stage="supervisor_rerank", source="Supervisor",
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            parsed = self._safe_parse_json(completion.choices[0].message.content.strip(), "Metric-Rerank")
            selected_names = parsed.get("selected_names", [])
            if not isinstance(selected_names, list):
                return fallback
            by_name = {metric.get("name"): metric for metric in candidates}
            selected = []
            for name in selected_names:
                metric = by_name.get(str(name))
                if metric and metric not in selected:
                    selected.append(metric)
                if len(selected) >= self.select_q:
                    break
            if self.exact_select_q and len(selected) < min(self.select_q, len(candidates)):
                for metric in candidates:
                    if metric not in selected:
                        selected.append(metric)
                    if len(selected) >= min(self.select_q, len(candidates)):
                        break
            return selected or fallback
        except Exception as e:
            print(f"[Supervisor Warning] Rerank failed, using top candidates: {e}")
            return fallback

    async def _match_metrics(self, task: str, output: str, metric_names=None) -> List[Dict]:
        if metric_names is not None:
            return [m for m in self.metrics if m['name'] in metric_names]

        if self.retrieval_mode == "random":
            if self.random_k_max > 0:
                low = self.random_k_min if self.random_k_min > 0 else 1
                high = max(low, self.random_k_max)
                k = random.randint(low, high)
            else:
                k = self.random_k
            print(f"[Supervisor] Mode: Random Selection (k={k})")
            k = min(k, len(self.metrics))
            return random.sample(self.metrics, k)

        summary_resp = await tracked_create(
            self._model_client.chat.completions.create, stage="supervisor_summary", source="Supervisor",
            model=self.model,
            messages=[{"role": "user", "content": SUMMARY_TEMPLATE.format(task=task, agent_output=output)}],
            temperature=0.0,
            max_tokens=1000,
        )
        raw_summary = summary_resp.choices[0].message.content.strip()
        try:
            json_match = re.search(r'\{.*\}', raw_summary, re.DOTALL)
            summary_data = json.loads(json_match.group(0))
            query_text = "Problem Scenario: " + ", ".join(summary_data.get("problem_scenario", [])) + \
                         ". Agent Action: " + ", ".join(summary_data.get("agent_action", []))
        except:
            query_text = raw_summary

        print(f"Generated Search Query: {query_text}")


        emb_resp = tracked_embedding_create(self.embedding_client.embeddings.create, stage="embedding_query", source="Supervisor", model=self.embedding_model, input=[query_text])
        query_emb = np.array(emb_resp.data[0].embedding, dtype=np.float32)
        if self.retrieval_mode == "rerank":
            print(f"[Supervisor] Mode: Rerank Search (Top-{self.retrieve_p} -> <=Top-{self.select_q}).")
            candidates = self._top_metric_candidates(query_emb, self.retrieve_p)
            return await self._rerank_metrics(task, output, candidates)

        print(f"[Supervisor] Mode: Direct Search (Top-{self.direct_k}).")
        return self._top_metric_candidates(query_emb, self.direct_k)
   
    async def _calc_score(self, task, message: TextMessage | dict, role: str = "Unknown", metrics=None) -> List[Dict]: 
        judgements = []
        message_content = message.content if isinstance(message, TextMessage) else message['content']
        
        if metrics is not None:
            matched_metrics = metrics
        else:
            matched_metrics = await self._match_metrics(task, message_content)

        if self.batch_audit_metrics and len(matched_metrics) > 1:
            return await self._calc_score_batch(task, message_content, role, matched_metrics)
        
        if self.retrieval_mode == "random":
            if self.domain == "code": target_template = METRIC_TEMPLATE_RANDOM_CODE
            else: target_template = METRIC_TEMPLATE_RANDOM_MATH
        elif self.domain == "code":
            target_template = METRIC_TEMPLATE_CODE_AUDIT
        else:
            target_template = METRIC_TEMPLATE_MATH_AUDIT
        
        for metric in matched_metrics:
            m_eval = metric.get('evaluator_prompt', {})
            trigger = m_eval.get('trigger_condition', 'N/A')
            risk_alert = m_eval.get('risk_alert', '') 
            audit_context = f"Context: {trigger}\nSpecific Risk: {risk_alert}"
            
            prompt = target_template.format(
                task=task,
                role=role,   
                agent_output=message_content,
                trigger_condition=audit_context 
            )
            
           
            for attempt in range(5):
                try:
                    completion = await tracked_create(
                        self._model_client.chat.completions.create, stage="supervisor_audit", source="Supervisor", metadata={"audit_attempt": attempt + 1},
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1500
                    )
                    res_raw = completion.choices[0].message.content.strip()
                    finding = self._safe_parse_json(res_raw, source_stage=f"Audit-{metric['name']}")
                    judgements.append(self._judgement_from_finding(metric['name'], finding))
                    break 

                except Exception as e:
                    if attempt == 4: 
                        print(f"[Audit Fail] Metric '{metric['name']}' failed 5 times.")
        
        return judgements

    async def _calc_score_batch(
        self,
        task: str,
        message_content: str,
        role: str,
        matched_metrics: List[Dict],
    ) -> List[Dict]:
        indicator_block = self._format_metric_candidates(matched_metrics, include_risk=True)
        prompt = BATCH_METRIC_TEMPLATE.format(
            domain=self.domain,
            task=task,
            role=role,
            agent_output=message_content,
            indicator_block=indicator_block,
        )

        for attempt in range(5):
            try:
                completion = await tracked_create(
                    self._model_client.chat.completions.create, stage="supervisor_audit_batch", source="Supervisor", metadata={"audit_attempt": attempt + 1},
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=3000,
                )
                parsed = self._safe_parse_json_any(
                    completion.choices[0].message.content.strip(),
                    "Batch-Audit",
                )
                if isinstance(parsed, dict):
                    parsed_items = parsed.get("judgements") or parsed.get("results") or []
                else:
                    parsed_items = parsed
                if not isinstance(parsed_items, list):
                    raise ValueError("Batch audit response is not a list.")

                by_metric = {}
                for item in parsed_items:
                    if not isinstance(item, dict):
                        continue
                    metric_name = item.get("metric") or item.get("metric_name") or item.get("name")
                    if metric_name:
                        by_metric[str(metric_name)] = item

                judgements = []
                for metric in matched_metrics:
                    metric_name = metric["name"]
                    finding = by_metric.get(metric_name)
                    if finding is None:
                        finding = {
                            "evidence_quote": "N/A",
                            "analysis": "No issue reported for this indicator.",
                            "suggestion": "N/A",
                            "impact_assessment": "NO",
                            "is_flawed": False,
                        }
                    judgements.append(self._judgement_from_finding(metric_name, finding))
                return judgements
            except Exception as e:
                if attempt == 4:
                    print(f"[Audit Fail] Batch audit failed 5 times: {e}")
        return []
    
  
    def update_scoreboard_with_results(self, message: TextMessage, judgements: list[dict]):
        if not self.prune_flag:
            self.scoreboard[message.id] = {
                "message": message,
                "judgements": judgements, 
                "is_pruned": False        
            }
            return

        pass_cnt = 0
        for judge in judgements:
            if judge['verdict'].lower() == 'correct':
                pass_cnt += 1
        
        is_pruned = (pass_cnt / len(judgements)) < self.pass_rate if judgements else False

        self.scoreboard[message.id] = {
            "message": message,
            "judgements": judgements,
            "is_pruned": is_pruned
        }
        
    async def judge(self, task: str, message: TextMessage, attempt_num: int, role: str = "Assistant", previous_metrics=None, session_metrics=None):
        if not self.prune_flag:
            return True, [], None, None 

        print("\n" + "="*80)
        print(f"--- round: {attempt_num} | Agent: {message.source} (Role: {role}) ---")
        print("="*80)
        
        current_metrics = []

   
        if self.use_simple_audit:
            print("[Supervisor] Using FIXED General Metric (Simple Audit Mode).")
        
            if "code" in self.domain or "mbpp" in self.domain:
                current_metrics = [SIMPLE_CODE_METRIC_1] 
            else:
                current_metrics = [SIMPLE_MATH_METRIC_1]
            
        else:
            
            print("[Supervisor] Retrieving NEW metrics from Pool...")
            message_content = message.content if isinstance(message, TextMessage) else message['content']
            current_metrics = await self._match_metrics(task, message_content)


        judgements = await self._calc_score(task, message, role=role, metrics=current_metrics)
        

        print(" -> " + ", ".join([j['metric'] for j in judgements]) if judgements else " ")


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
                
            
                print(f"   - : {j.get('evidence_quote', 'N/A')}")
                print(f"   - : {j.get('reasoning', 'N/A')}")
                print(f"   - : {j.get('suggestion', 'N/A')}")
                print(f"   - : {j.get('impact', 'N/A')}")
                
           
                reason = j.get('reasoning', 'N/A')
                short_reason = (reason[:1000] + '...') if len(reason) > 1000 else reason
                
                item = (
                    f"- [{metric_name}]: {j.get('suggestion', 'N/A')}\n"
                    f"  (Auditor's Note: {short_reason})"
                )
                feedback_lines.append(item)
        
        total_metrics = len(judgements)
        pass_flag = (pass_cnt / total_metrics) >= self.pass_rate if total_metrics > 0 else True 
        
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
            
        print("\n[3]judge:")
        print(f" -> pass rate: {pass_cnt}/{total_metrics} | threshold: {self.pass_rate:.0%} | result: {'pass' if pass_flag else 'fail'}")
        
        if feedback:
            print("\n[4] feedback:")
            print(feedback)
            
        print("="*80 + "\n")
        
        return pass_flag, judgements, feedback, current_metrics
    
    def prune_info(self, all_messages: List[TextMessage]) -> List[TextMessage]:
        if not self.prune_flag: return all_messages
        pruned_messages = []
        for msg in all_messages:
            if msg.id in self.scoreboard:
                if not self.scoreboard[msg.id]['is_pruned']:
                    pruned_messages.append(msg)
                else:
                    # print(f"Info: Message ID {msg.id} filtered out by supervisor.")
                    pass
            else:
                if msg.source != "user":
                    pruned_messages.append(msg)
        return pruned_messages
    
    def get_messages_above_threshold(self) -> List[TextMessage]:
        if not self.prune_flag:
            return [entry["message"] for entry in self.scoreboard.values()]
        return [entry["message"] for entry in self.scoreboard.values() if not entry["is_pruned"]]
    
    def get_scores(self, role_map):
        ret_scores = {}
        for entry in self.scoreboard:
            ret_scores[entry] = {
                'role': role_map[self.scoreboard[entry]["message"].source],
                'message': self.scoreboard[entry]['message'].dump(),
                'judgements': self.scoreboard[entry]['judgements'],
                'is_pruned': self.scoreboard[entry]['is_pruned'],
            }
        return ret_scores
    
    def reset(self):
        self.scoreboard = {}
