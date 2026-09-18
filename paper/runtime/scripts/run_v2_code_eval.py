from __future__ import annotations

import argparse
import asyncio
import builtins
import contextvars
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from openai import DefaultAsyncHttpxClient, Timeout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from AgentDropout.agents import AgentRegistry
from AgentDropout.agents.agent_mode_helpers import (
    attach_agent_mode_state,
    configure_agent_mode_team,
    get_agent_mode_history,
    normalize_agent_mode,
    reset_agent_mode_state,
)
from AgentDropout.agents.supervisor_reasoning_pick_metric import Supervisor
from AgentDropout.usage_tracking import TrackedOpenAIChatCompletionClient
from AgentDropout.tools.coding.python_executor import HumanEvalExecutor, MBPPExecutor
from v2_code_common import ensure_task_name


_current_log_buffer = contextvars.ContextVar("current_log_buffer", default=None)
_global_log_store: dict[str, str] = {}
_original_print = builtins.print
_file_write_lock = asyncio.Lock()


def scoped_print(*args, **kwargs):
    buffer = _current_log_buffer.get()
    if buffer:
        kwargs["file"] = buffer
        _original_print(*args, **kwargs)
    else:
        _original_print(*args, **kwargs)


def load_global_resources(metric_file: str, cache_file: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    print("Loading Global Resources...")
    print(f" - Metrics: {metric_file}")
    print(f" - Cache:   {cache_file}")

    with open(metric_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    emb_map: dict[str, list[float]] = {}
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            emb_map[record["name"]] = record["vector"]

    vectors_list = []
    vector_dim = len(next(iter(emb_map.values()))) if emb_map else 1024
    for metric in metrics:
        name = metric["name"]
        vector = emb_map.get(name)
        if vector is None:
            vectors_list.append(np.zeros(vector_dim, dtype=np.float32))
        else:
            vectors_list.append(np.array(vector, dtype=np.float32))

    embeddings = np.stack(vectors_list) if vectors_list else np.zeros((0, vector_dim), dtype=np.float32)
    print(f"Global Resources Loaded. Embedding Shape: {embeddings.shape}")
    return metrics, embeddings


def extract_code_from_response(response: str) -> str:
    if not isinstance(response, str):
        return ""
    content = response.strip()
    if not content:
        return ""

    python_blocks = re.findall(r"```python\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if python_blocks:
        for block in python_blocks:
            candidate = block.strip()
            if candidate:
                return candidate

    generic_blocks = re.findall(r"```\s*(.*?)\s*```", content, re.DOTALL)
    if generic_blocks:
        for block in generic_blocks:
            candidate = block.strip()
            if candidate:
                return candidate

    if (
        content.startswith("def ")
        or content.startswith("class ")
        or content.startswith("import ")
        or content.startswith("from ")
    ):
        return content

    return content if "if __name__" in content else ""


def prepare_humaneval_data(item: dict[str, Any]) -> dict[str, Any] | None:
    prompt = item.get("prompt")
    if not prompt:
        return None
    task_id = item.get("task_id", item.get("name", "unknown"))
    return {
        "id": str(task_id),
        "task_prompt": prompt,
        "test": item.get("test", ""),
        "entry_point": item.get("entry_point", ""),
    }


def prepare_mbpp_prompt_data(item: dict[str, Any]) -> dict[str, Any] | None:
    task_description = item.get("text") or item.get("prompt")
    code_string = item.get("code")
    test_list = item.get("test_list", [])
    if not task_description or not code_string or not test_list:
        return None

    first_test = test_list[0]
    match_test = re.search(r"assert\s+([a-zA-Z_]\w*)\s*\(", first_test)
    if match_test:
        entry_point = match_test.group(1)
    else:
        matches = list(re.finditer(r"def\s+([a-zA-Z_]\w*)\s*\(", code_string))
        if not matches:
            return None
        entry_point = matches[-1].group(1)

    sig_pattern = rf"def\s+{re.escape(entry_point)}\s*\(.*?\)\s*:"
    match_sig = re.search(sig_pattern, code_string, re.DOTALL)
    prompt_suffix = match_sig.group(0) if match_sig else f"def {entry_point}("
    prompt_text = f"{task_description.strip()}\n\n{prompt_suffix}"

    return {
        "id": str(item.get("task_id", "unknown")),
        "task_prompt": prompt_text,
        "test": "\n".join(test_list),
        "entry_point": entry_point,
    }


def _generate_short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def prepare_codecontest_data(item: dict[str, Any], idx: int) -> dict[str, Any] | None:
    problem = item.get("problem", "")
    tests = item.get("tests", {})
    if not problem or not isinstance(tests, dict):
        return None

    task_id = item.get("id", item.get("name", str(idx)))
    if str(task_id).isdigit():
        task_id = f"{task_id}_{_generate_short_hash(problem)}"

    return {
        "id": str(task_id),
        "task_prompt": problem,
        "tests": tests,
    }


def prepare_livecode_data(item: dict[str, Any]) -> dict[str, Any] | None:
    problem = item.get("question_content", "")
    if not problem:
        return None

    starter_code = item.get("starter_code", "")
    public_tests_raw = item.get("public_test_cases", "[]")
    test_cases: list[dict[str, Any]] = []
    try:
        if isinstance(public_tests_raw, str):
            test_cases = json.loads(public_tests_raw)
        elif isinstance(public_tests_raw, list):
            test_cases = public_tests_raw
    except Exception:
        test_cases = []

    valid_tests = [
        test_case
        for test_case in test_cases
        if isinstance(test_case, dict) and "input" in test_case and "output" in test_case
    ]

    task_prompt = problem
    if starter_code:
        task_prompt += f"\n\nStarter Code:\n```python\n{starter_code}\n```"

    return {
        "id": str(item.get("question_id", item.get("id", "unknown"))),
        "task_prompt": task_prompt,
        "starter_code": starter_code,
        "test_cases": valid_tests,
        "platform": item.get("platform", "unknown"),
        "difficulty": item.get("difficulty", "unknown"),
    }


def _parse_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value is not None else default


def prepare_apps_data(item: dict[str, Any], idx: int) -> dict[str, Any] | None:
    question = item.get("question") or item.get("problem") or ""
    raw_io = _parse_json_field(item.get("input_output"), {})
    if not question:
        return None
    if not isinstance(raw_io, dict):
        raw_io = {}

    fn_name = raw_io.get("fn_name")
    if fn_name:
        format_instruction = (
            "\n\nAPPS/TACO call-based format requirements:\n"
            f"- Implement exactly the Python function `{fn_name}` with the signature implied by the problem.\n"
            f"- Define `{fn_name}` as a top-level function, for example `def {fn_name}(...):`.\n"
            "- Do not wrap the function in `class Solution` or any other class.\n"
            "- Avoid undefined type annotations such as `List`; either omit annotations or import required names.\n"
            "- Return the required value from that function.\n"
            "- Do not read from stdin, do not call input(), and do not print the answer.\n"
            "- The final answer must include a Python code block containing the function definition."
        )
    else:
        format_instruction = (
            "\n\nAPPS standard-input format: write a complete Python 3 script that reads from stdin "
            "and prints the required answer to stdout."
        )
    starter_code = item.get("starter_code") or ""
    starter_block = ""
    if starter_code:
        starter_block = f"\n\nStarter Code:\n```python\n{starter_code}\n```"

    task_id = item.get("problem_id", item.get("id", idx))
    prepared = {
        "id": f"apps_{task_id}",
        "task_prompt": question + starter_block + format_instruction,
        "input_output": raw_io,
        "fn_name": fn_name,
        "difficulty": item.get("difficulty", "unknown"),
        "url": item.get("url", ""),
        "starter_code": starter_code,
        "apps_split": item.get("apps_split", item.get("split", "")),
        "apps_config": item.get("apps_config", item.get("config", "")),
    }
    for key in (
        "taco_split",
        "taco_source",
        "taco_raw_tags",
        "taco_tags",
        "taco_skill_types",
        "taco_time_limit",
        "taco_memory_limit",
        "taco_expected_time_complexity",
        "taco_expected_auxiliary_space",
        "source",
        "tags",
        "skill_types",
    ):
        if key in item:
            prepared[key] = item[key]
    return prepared


def iter_raw_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        for key in ("data", "items", "examples", "records", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        for item in payload.values():
            if isinstance(item, dict):
                yield item


def iter_prepared_instances(spec, input_path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    count = 0
    for idx, item in enumerate(iter_raw_records(input_path)):
        if spec.dataset_format == "humaneval":
            prepared = prepare_humaneval_data(item)
        elif spec.dataset_format == "mbpp":
            prepared = prepare_mbpp_prompt_data(item)
        elif spec.dataset_format == "codecontest":
            prepared = prepare_codecontest_data(item, idx)
        elif spec.dataset_format == "livecode":
            prepared = prepare_livecode_data(item)
        elif spec.dataset_format == "apps":
            prepared = prepare_apps_data(item, idx)
        else:
            raise ValueError(f"Unsupported dataset format: {spec.dataset_format}")

        if prepared is None:
            continue

        yield prepared
        count += 1
        if limit is not None and count >= limit:
            break


class CodeContestExecutor:
    def __init__(self, timeout: int = 2):
        self.timeout = timeout

    @staticmethod
    def normalize(output: str) -> str:
        if not output:
            return ""
        lines = output.strip().splitlines()
        return "\n".join(line.rstrip() for line in lines)

    def execute(self, code: str, tests: dict[str, list[str]]) -> tuple[bool, str, str]:
        if not code:
            return False, "No code generated", ""

        inputs = tests.get("inputs", [])
        outputs = tests.get("outputs", [])
        if not inputs or not outputs:
            return False, "No test cases found", ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            tmp_file_path = tmp_file.name

        passed_count = 0
        total_count = len(inputs)
        first_error = ""

        try:
            for i in range(total_count):
                try:
                    process = subprocess.run(
                        [sys.executable, tmp_file_path],
                        input=inputs[i].encode("utf-8"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout,
                    )
                    if process.returncode != 0:
                        err = process.stderr.decode("utf-8", errors="ignore").strip()
                        if not first_error:
                            first_error = f"Runtime Error on case {i}: {err}"
                        continue

                    actual_out = process.stdout.decode("utf-8", errors="ignore")
                    if self.normalize(actual_out) == self.normalize(outputs[i]):
                        passed_count += 1
                    elif not first_error:
                        short_act = actual_out[:100] + "..." if len(actual_out) > 100 else actual_out
                        first_error = f"Wrong Answer on case {i}. Got: {short_act}"

                except subprocess.TimeoutExpired:
                    if not first_error:
                        first_error = f"Time Limit Exceeded on case {i}"

            is_solved = passed_count == total_count
            return is_solved, f"Passed {passed_count}/{total_count} cases", first_error
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


class LiveCodeExecutor:
    def __init__(self, timeout: int = 4):
        self.timeout = timeout

    @staticmethod
    def normalize(output: str) -> str:
        if not output:
            return ""
        lines = output.strip().splitlines()
        return "\n".join(line.rstrip() for line in lines)

    def execute(self, code: str, test_cases: list[dict[str, Any]]) -> tuple[bool, str, str]:
        if not code:
            return False, "No code generated", ""
        if not test_cases:
            return False, "No public test cases found", ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            tmp_file_path = tmp_file.name

        passed_count = 0
        total_count = len(test_cases)
        first_error = ""

        try:
            for i, case in enumerate(test_cases):
                try:
                    process = subprocess.run(
                        [sys.executable, tmp_file_path],
                        input=case.get("input", "").encode("utf-8"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout,
                    )
                    if process.returncode != 0:
                        err = process.stderr.decode("utf-8", errors="ignore").strip()
                        if not first_error:
                            first_error = f"Runtime Error on case {i}: {err}"
                        continue

                    actual_out = process.stdout.decode("utf-8", errors="ignore")
                    expected_out = case.get("output", "")
                    if self.normalize(actual_out) == self.normalize(expected_out):
                        passed_count += 1
                    elif not first_error:
                        short_act = actual_out[:100] + "..." if len(actual_out) > 100 else actual_out
                        first_error = f"Wrong Answer on case {i}. Got: {short_act}"

                except subprocess.TimeoutExpired:
                    if not first_error:
                        first_error = f"TLE on case {i}"

            is_solved = passed_count == total_count
            return is_solved, f"Passed {passed_count}/{total_count} public cases", first_error
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


class AppsExecutor:
    def __init__(self, timeout: int = 4):
        self.timeout = timeout

    @staticmethod
    def normalize(output: str) -> str:
        if not output:
            return ""
        lines = output.strip().splitlines()
        return "\n".join(line.rstrip() for line in lines)

    @staticmethod
    def _values_match(actual: Any, expected: Any) -> bool:
        if actual == expected:
            return True
        if isinstance(actual, tuple) and list(actual) == expected:
            return True
        if isinstance(expected, tuple) and actual == list(expected):
            return True
        return AppsExecutor.normalize(str(actual)) == AppsExecutor.normalize(str(expected))

    def _execute_stdio(self, code: str, inputs: list[Any], outputs: list[Any]) -> tuple[bool, str, str]:
        tests = {
            "inputs": [item if isinstance(item, str) else str(item) for item in inputs],
            "outputs": [item if isinstance(item, str) else str(item) for item in outputs],
        }
        return CodeContestExecutor(timeout=self.timeout).execute(code, tests)

    def _execute_call_based(
        self,
        code: str,
        fn_name: str,
        inputs: list[Any],
        outputs: list[Any],
    ) -> tuple[bool, str, str]:
        if not code:
            return False, "No code generated", ""

        harness = """
import contextlib
import io
import json
import signal
import sys

payload = json.loads(sys.stdin.read())
code = payload["code"]
fn_name = payload["fn_name"]
case_input = payload["input"]
expected = payload["expected"]
timeout = int(payload.get("timeout", 4))

def _timeout(_signum, _frame):
    raise TimeoutError("function timeout")

ns = {"__name__": "not_main"}
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(timeout)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "<apps_solution>", "exec"), ns)
    fn = ns[fn_name]
    if isinstance(case_input, list):
        actual = fn(*case_input)
    else:
        actual = fn(case_input)
    signal.alarm(0)
    print(json.dumps({"ok": True, "actual": actual, "expected": expected}, ensure_ascii=False))
except Exception as exc:
    signal.alarm(0)
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(harness)
            tmp_file_path = tmp_file.name

        passed_count = 0
        total_count = min(len(inputs), len(outputs))
        first_error = ""
        try:
            for i in range(total_count):
                payload = {
                    "code": code,
                    "fn_name": fn_name,
                    "input": inputs[i],
                    "expected": outputs[i],
                    "timeout": self.timeout,
                }
                try:
                    process = subprocess.run(
                        [sys.executable, tmp_file_path],
                        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout + 1,
                    )
                except subprocess.TimeoutExpired:
                    if not first_error:
                        first_error = f"Time Limit Exceeded on case {i}"
                    continue

                if process.returncode != 0:
                    err = process.stderr.decode("utf-8", errors="ignore").strip()
                    if not first_error:
                        first_error = f"Runtime Error on case {i}: {err}"
                    continue

                try:
                    result = json.loads(process.stdout.decode("utf-8", errors="ignore"))
                except Exception:
                    if not first_error:
                        first_error = f"Invalid harness output on case {i}"
                    continue

                if not result.get("ok"):
                    if not first_error:
                        first_error = f"Runtime Error on case {i}: {result.get('error', '')}"
                    continue

                if self._values_match(result.get("actual"), result.get("expected")):
                    passed_count += 1
                elif not first_error:
                    first_error = f"Wrong Answer on case {i}. Got: {result.get('actual')}"

            is_solved = passed_count == total_count and total_count > 0
            return is_solved, f"Passed {passed_count}/{total_count} cases", first_error
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def execute(self, code: str, input_output: dict[str, Any]) -> tuple[bool, str, str]:
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        fn_name = input_output.get("fn_name")
        if not inputs or not outputs:
            return False, "No test cases found", ""
        if fn_name:
            return self._execute_call_based(code, str(fn_name), inputs, outputs)
        return self._execute_stdio(code, inputs, outputs)


def init_team(spec, preloaded_metrics, preloaded_embeddings):
    use_llm = not args.force_direct_search
    supervisor = Supervisor(
        model=args.supervisor_model,
        api_key=os.environ.get("AGENTDROPOUT_SUPERVISOR_API_KEY", "EMPTY"),
        base_url=args.supervisor_url,
        domain="code",
        metrics_retrieve_k=args.retrieve_p,
        pass_rate=args.pass_rate,
        prune_flag=True,
        metric_pool_file=args.metric_pool_file,
        embedding_cache_file=args.embedding_cache_file,
        embedding_api_key=os.environ.get("AGENTDROPOUT_EMBEDDING_API_KEY", "EMPTY"),
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_url,
        preloaded_metrics=preloaded_metrics,
        preloaded_embeddings=preloaded_embeddings,
        use_llm_rerank=use_llm,
        max_metrics_count=args.select_q,
        force_direct_search=args.force_direct_search,
        direct_k=args.direct_k,
        retrieve_p=args.retrieve_p,
        select_q=args.select_q,
        random_k=args.random_k,
        use_simple_audit=args.use_simple_audit,
    )

    agent_registry = AgentRegistry()
    participants = [
        agent_registry.get(
            agent_name=spec.agent_name,
            name=f"Participant_{i + 1}",
            domain=spec.domain_name,
            model=args.reasoning_model,
            api_key=os.environ.get("AGENTDROPOUT_REASONING_API_KEY", "EMPTY"),
            base_url=args.reasoning_url,
            supervisor=supervisor,
        )
        for i in range(5)
    ]

    role_map = {agent.name: agent.role for agent in participants}
    for agent in participants:
        agent.role_map = role_map
        if not hasattr(agent, "description"):
            agent.description = f"An AI agent with the role of {agent.role}."

    shared_score_board, predictor = configure_agent_mode_team(
        participants,
        agent_mode=args.agent_mode,
        prm_url=args.prm_url,
        prm_n_samples=args.prm_n_samples,
        model=args.reasoning_model,
        api_key=os.environ.get("AGENTDROPOUT_REASONING_API_KEY", "EMPTY"),
        base_url=args.reasoning_url,
    )

    selector_prompt = """Select an agent to perform task.
{roles}
Current conversation context:
{history}
Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

    model_client = TrackedOpenAIChatCompletionClient(
        model=args.selector_model,
        api_key=args.selector_api_key,
        base_url=args.selector_url,
        http_client=DefaultAsyncHttpxClient(
            trust_env=False,
            timeout=Timeout(120.0, connect=10.0),
        ),
        timeout=Timeout(120.0, connect=10.0),
        max_retries=5,
        usage_stage="selector",
        usage_source="SelectorGroupChat",
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=args.max_turns)

    team = SelectorGroupChat(
        participants=participants,
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,
    )
    attach_agent_mode_state(team, shared_score_board, predictor)

    decision_maker = AgentRegistry.get(
        agent_name=spec.decision_agent_name,
        name="DecisionMaker",
        domain=spec.domain_name,
        model=args.reasoning_model,
        api_key=os.environ.get("AGENTDROPOUT_REASONING_API_KEY", "EMPTY"),
        base_url=args.reasoning_url,
    )

    return team, decision_maker, role_map, supervisor


async def reasoning(task_prompt: str, team, decision_maker, role_map, supervisor):
    is_baseline_mode = getattr(args, "baseline_only", False)
    agent_mode = normalize_agent_mode(getattr(args, "agent_mode", "supervisor"))

    if agent_mode != "supervisor":
        print(f"\n>>> [Mode] Code agent comparison mode: {agent_mode}")
        await team.reset()
        reset_agent_mode_state(team)
        supervisor.reset()
        supervisor.prune_flag = False
        await Console(team.run_stream(task=task_prompt))
        history_messages = get_agent_mode_history(team)
        print(f"[Agent Mode Result] collected messages: {len(history_messages)}")
        if not history_messages:
            history_messages = supervisor.get_messages_above_threshold()
    elif is_baseline_mode:
        print("\n>>> [Mode] Baseline Only (No Audit / No Pruning)")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = False
        await Console(team.run_stream(task=task_prompt))
        history_messages = supervisor.get_messages_above_threshold()
    else:
        print("\n>>> [Phase 1] 启动 Code Pool Audit 模式")
        await team.reset()
        supervisor.reset()
        supervisor.prune_flag = True
        await Console(team.run_stream(task=task_prompt))

        history_messages = supervisor.get_messages_above_threshold()
        retained_count = len(history_messages)
        print(f"[Check] 本轮保留消息数: {retained_count}")

        if retained_count <= 1:
            print("⚠️ 触发保底机制，重新执行无剪枝版本。")
            await team.reset()
            supervisor.reset()
            original_prune_flag = supervisor.prune_flag
            supervisor.prune_flag = False
            await Console(team.run_stream(task=task_prompt))
            history_messages = supervisor.get_messages_above_threshold()
            supervisor.prune_flag = original_prune_flag

    raw_answer = await decision_maker.run_decision(
        history_messages=history_messages,
        role_map=role_map,
        task=task_prompt,
    )
    return raw_answer.content.strip(), supervisor.get_scores(role_map), getattr(supervisor, "reflection_records", [])


def evaluate_instance(spec, code: str, instance: dict[str, Any]) -> tuple[bool, str, str]:
    if not code:
        return False, "No code extracted", ""

    if spec.dataset_format == "mbpp":
        is_solved, feedback, _ = MBPPExecutor().execute(code, [instance["test"]], timeout=15)
        return bool(is_solved), feedback, ""

    if spec.dataset_format == "humaneval":
        is_solved, feedback, _ = HumanEvalExecutor().execute(
            code,
            [instance["test"]],
            entry_point=instance.get("entry_point"),
            timeout=15,
        )
        return bool(is_solved), feedback, ""

    if spec.dataset_format == "codecontest":
        return CodeContestExecutor(timeout=2).execute(code, instance["tests"])

    if spec.dataset_format == "livecode":
        return LiveCodeExecutor(timeout=4).execute(code, instance["test_cases"])

    if spec.dataset_format == "apps":
        return AppsExecutor(timeout=4).execute(code, instance["input_output"])

    raise ValueError(f"Unsupported dataset format: {spec.dataset_format}")


async def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    async with _file_write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def run_sample(spec, instance: dict[str, Any], out_file: Path, preloaded_metrics, preloaded_embeddings) -> None:
    instance_id = str(instance.get("id", "unknown"))
    use_buffer = not args.disable_log_buffer
    log_capture = None
    token = None
    if use_buffer:
        log_capture = io.StringIO()
        token = _current_log_buffer.set(log_capture)

    try:
        print(f"--- [ {time.strftime('%Y-%m-%d %H:%M:%S')} ] ---")
        print(f"开始处理: {instance_id}")

        team, decision_maker, role_map, supervisor = init_team(
            spec,
            preloaded_metrics,
            preloaded_embeddings,
        )
        raw_content, scores, reflection_records = await reasoning(
            instance["task_prompt"],
            team,
            decision_maker,
            role_map,
            supervisor,
        )

        answer_code = extract_code_from_response(raw_content)
        is_solved, execution_result, error_text = evaluate_instance(spec, answer_code, instance)
        print(f"完成处理: {instance_id} | Solved: {is_solved}")

        record = {
            "id": instance_id,
            "task": spec.name,
            "task_prompt": instance["task_prompt"],
            "hypothesis": answer_code,
            "raw_response": raw_content,
            "is_solved": bool(is_solved),
            "execution_result": execution_result,
            "error": error_text,
            "scores": scores,
            "reflection_records": reflection_records,
        }
        if "entry_point" in instance:
            record["entry_point"] = instance["entry_point"]
        if "platform" in instance:
            record["platform"] = instance["platform"]
        if "difficulty" in instance:
            record["difficulty"] = instance["difficulty"]
        if "url" in instance:
            record["url"] = instance["url"]
        if "fn_name" in instance:
            record["fn_name"] = instance["fn_name"]
        if "starter_code" in instance:
            record["starter_code"] = instance["starter_code"]
        if "apps_split" in instance:
            record["apps_split"] = instance["apps_split"]
        if "apps_config" in instance:
            record["apps_config"] = instance["apps_config"]
        for key in (
            "taco_split",
            "taco_source",
            "taco_raw_tags",
            "taco_tags",
            "taco_skill_types",
            "taco_time_limit",
            "taco_memory_limit",
            "taco_expected_time_complexity",
            "taco_expected_auxiliary_space",
        ):
            if key in instance:
                record[key] = instance[key]

        await append_jsonl(out_file, record)

    except Exception as exc:
        _original_print(f"!!!!!! [CRITICAL ERROR] Task {instance_id}: {exc} !!!!!!")
        _original_print(traceback.format_exc())
        print(f"Error processing task {instance_id}: {exc}")
        traceback.print_exc()
    finally:
        if use_buffer and log_capture is not None and token is not None:
            _global_log_store[instance_id] = log_capture.getvalue()
            log_capture.close()
            _current_log_buffer.reset(token)


async def main() -> int:
    if not args.disable_log_buffer:
        builtins.print = scoped_print

    spec = ensure_task_name(args.task)
    input_path = Path(args.in_file)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1

    out_file = Path(args.out_file)
    if out_file.exists():
        out_file.unlink()

    global_metrics, global_embeddings = load_global_resources(
        args.metric_pool_file,
        args.embedding_cache_file,
    )

    if args.log_file:
        final_log_file = Path(args.log_file)
    else:
        final_log_file = out_file.with_name(out_file.stem + "_full.log")
    final_log_file.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency_limit)
    pending: set[asyncio.Task[Any]] = set()
    scheduled = 0

    async def worker(prepared_instance: dict[str, Any]) -> None:
        async with semaphore:
            sample_timeout = float(args.sample_timeout_seconds or 0)
            try:
                if sample_timeout > 0:
                    await asyncio.wait_for(
                        run_sample(
                            spec,
                            prepared_instance,
                            out_file,
                            global_metrics,
                            global_embeddings,
                        ),
                        timeout=sample_timeout,
                    )
                else:
                    await run_sample(
                        spec,
                        prepared_instance,
                        out_file,
                        global_metrics,
                        global_embeddings,
                    )
            except asyncio.TimeoutError:
                instance_id = str(prepared_instance.get("id", "unknown"))
                print(f"[TIMEOUT] Task {instance_id} exceeded {sample_timeout:.1f}s; writing failed record.")
                record = {
                    "id": instance_id,
                    "task": spec.name,
                    "task_prompt": prepared_instance.get("task_prompt", ""),
                    "hypothesis": "",
                    "raw_response": "",
                    "is_solved": False,
                    "execution_result": f"sample_timeout_seconds={sample_timeout:.1f}",
                    "error": "sample_timeout",
                    "scores": {},
                    "reflection_records": [],
                }
                for key in (
                    "entry_point",
                    "platform",
                    "difficulty",
                    "url",
                    "fn_name",
                    "starter_code",
                    "apps_split",
                    "apps_config",
                ):
                    if key in prepared_instance:
                        record[key] = prepared_instance[key]
                await append_jsonl(out_file, record)

    start_time = time.time()

    for prepared_instance in iter_prepared_instances(spec, input_path, args.limit):
        scheduled += 1
        pending.add(asyncio.create_task(worker(prepared_instance)))

        if scheduled % 50 == 0:
            print(f"[Progress] 已调度 {scheduled} 个任务")

        if len(pending) >= max(1, args.concurrency_limit * 2):
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task

    if pending:
        await asyncio.gather(*pending)

    total_time = time.time() - start_time
    print(f"\n🎉 任务完成! 总耗时: {total_time:.2f}s | 总任务数: {scheduled}")

    if scheduled == 0:
        print("[ERROR] No tasks were scheduled.")
        return 1

    if not args.disable_log_buffer:
        def sort_key(key: str):
            try:
                return int(key)
            except Exception:
                return key

        with final_log_file.open("w", encoding="utf-8") as f:
            f.write(f"=== {spec.name} v2 code logs ===\n")
            f.write(f"Total tasks: {scheduled}\n")
            f.write(f"Total time: {total_time:.2f}s\n\n")
            for task_id in sorted(_global_log_store.keys(), key=sort_key):
                f.write(f"\n{'=' * 40}\n=== TASK ID: {task_id} ===\n{'=' * 40}\n")
                f.write(_global_log_store[task_id])
                f.write("\n")
        print(f"✅ Logs saved to: {final_log_file}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--in_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--reasoning_url", required=True)
    parser.add_argument("--reasoning_model", required=True)
    parser.add_argument("--supervisor_url", required=True)
    parser.add_argument("--supervisor_model", required=True)
    parser.add_argument("--embedding_url", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--selector_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--selector_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument(
        "--selector_api_key",
        type=str,
        default="EMPTY",
    )
    parser.add_argument("--metric_pool_file", required=True)
    parser.add_argument("--embedding_cache_file", required=True)
    parser.add_argument("--concurrency_limit", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=7)
    parser.add_argument("--pass_rate", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_log_buffer", action="store_true")
    parser.add_argument("--sample_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--use_simple_audit", type=int, default=0)
    parser.add_argument("--force_direct_search", action="store_true", default=False)
    parser.add_argument("--direct_k", type=int, default=5)
    parser.add_argument("--retrieve_p", type=int, default=20)
    parser.add_argument("--select_q", type=int, default=5)
    parser.add_argument("--random_k", type=int, default=0)
    parser.add_argument("--agent_mode", default="supervisor", choices=["supervisor", "prm", "self_refine", "self-refine", "multi_tag", "multi-tag"])
    parser.add_argument("--prm_url", type=str, default=None)
    parser.add_argument("--prm_n_samples", type=int, default=3)
    args = parser.parse_args()

    raise SystemExit(asyncio.run(main()))
