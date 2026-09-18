from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent / "assets"
REBUTTAL_ROOT = ROOT / "26.3.24rebuttal"
UPSTREAM_TEST_ROOT = REBUTTAL_ROOT / "AgentDropoutV2_upstream" / "test"


@dataclass(frozen=True)
class TaskSpec:
    task: str
    paper_name: str
    paper_score: float
    judge_note: str
    dataset_candidates: tuple[Path, ...]


TASK_SPECS = {
    "mbpp": TaskSpec(
        task="mbpp",
        paper_name="MBPP",
        paper_score=68.09,
        judge_note="MBPP assertion execution; timeout=15s",
        dataset_candidates=(
            UPSTREAM_TEST_ROOT / "project_datasets" / "mbpp-sanitized" / "mbpp_test.jsonl",
            ROOT
            / "AgentDropout_v2"
            / "selection_group_agentdropout"
            / "AutoGen-3"
            / "project_datasets"
            / "mbpp-sanitized"
            / "mbpp_test.jsonl",
        ),
    ),
    "humaneval": TaskSpec(
        task="humaneval",
        paper_name="HumanEval",
        paper_score=84.50,
        judge_note="HumanEval test execution; timeout=15s",
        dataset_candidates=(
            UPSTREAM_TEST_ROOT / "project_datasets" / "humaneval" / "humaneval-py_id.jsonl",
            ROOT / "AutoGen（train-mbpp）" / "project_datasets" / "humaneval" / "humaneval-py.jsonl",
        ),
    ),
    "codecontest": TaskSpec(
        task="codecontest",
        paper_name="CodeContests",
        paper_score=9.26,
        judge_note="Public stdin/stdout cases; exact normalized match; timeout=2s/case",
        dataset_candidates=(
            UPSTREAM_TEST_ROOT / "project_datasets" / "codecontest" / "test.jsonl",
            ROOT
            / "AgentDropout_v2"
            / "selection_group_agentdropout"
            / "AutoGen-3"
            / "project_datasets"
            / "codecontest"
            / "test.jsonl",
        ),
    ),
    "livecode": TaskSpec(
        task="livecode",
        paper_name="LiveCodeBench",
        paper_score=32.75,
        judge_note="Public test cases only; exact normalized match; timeout=4s/case",
        dataset_candidates=(
            UPSTREAM_TEST_ROOT / "project_datasets" / "livecode" / "livecodebench_v1.jsonl",
            ROOT
            / "AgentDropout_v2"
            / "selection_group_agentdropout"
            / "AutoGen-3"
            / "project_datasets"
            / "livecode"
            / "livecodebench_v1.jsonl",
        ),
    ),
}

TASK_ALIASES = {
    "human_eval": "humaneval",
    "human-eval": "humaneval",
    "codecontests": "codecontest",
    "code_contests": "codecontest",
    "code-contests": "codecontest",
    "livecodebench": "livecode",
    "live-code-bench": "livecode",
    "live_code_bench": "livecode",
}


def ensure_task_name(task: str) -> str:
    normalized = task.strip().casefold().replace("-", "_").replace(" ", "_")
    canonical = TASK_ALIASES.get(normalized, normalized)
    if canonical not in TASK_SPECS:
        raise KeyError(f"Unknown task: {task}")
    return canonical


def resolve_dataset_path(task: str, dataset_file: str | None) -> Path:
    if dataset_file:
        path = Path(dataset_file)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return path
    for candidate in TASK_SPECS[task].dataset_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No default dataset found for task={task}")


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


def _generate_short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def prepare_mbpp_data(item: dict[str, Any]) -> dict[str, Any] | None:
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

    return {
        "id": str(item.get("task_id", "unknown")),
        "entry_point": entry_point,
        "test": "\n".join(test_list),
        "task_prompt": task_description.strip(),
    }


def prepare_humaneval_data(item: dict[str, Any]) -> dict[str, Any] | None:
    prompt = item.get("prompt")
    if not prompt:
        return None
    return {
        "id": str(item.get("task_id", item.get("name", "unknown"))),
        "test": item.get("test", ""),
        "entry_point": item.get("entry_point", ""),
        "task_prompt": prompt,
    }


def prepare_codecontest_data(item: dict[str, Any], idx: int) -> dict[str, Any] | None:
    problem = item.get("problem") or item.get("description")
    tests = item.get("tests") or item.get("public_tests") or {}
    if not problem or not isinstance(tests, dict):
        return None

    raw_id = str(item.get("id", item.get("name", idx)))
    hashed_id = raw_id
    if raw_id.isdigit():
        hashed_id = f"{raw_id}_{_generate_short_hash(problem)}"

    return {
        "id": raw_id,
        "hashed_id": hashed_id,
        "tests": tests,
        "task_prompt": problem,
    }


def prepare_livecode_data(item: dict[str, Any]) -> dict[str, Any] | None:
    problem = item.get("question_content", "")
    if not problem:
        return None

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

    starter_code = item.get("starter_code", "")
    task_prompt = problem
    if starter_code:
        task_prompt += f"\n\nStarter Code:\n```python\n{starter_code}\n```"

    return {
        "id": str(item.get("question_id", item.get("id", "unknown"))),
        "test_cases": valid_tests,
        "task_prompt": task_prompt,
    }


def load_dataset_lookup(task: str, dataset_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_prompt: dict[str, dict[str, Any]] = {}

    for idx, item in enumerate(iter_raw_records(dataset_path)):
        if task == "mbpp":
            prepared = prepare_mbpp_data(item)
        elif task == "humaneval":
            prepared = prepare_humaneval_data(item)
        elif task == "codecontest":
            prepared = prepare_codecontest_data(item, idx)
        elif task == "livecode":
            prepared = prepare_livecode_data(item)
        else:
            raise ValueError(f"Unsupported task: {task}")

        if prepared is None:
            continue

        by_id[prepared["id"]] = prepared
        if task == "codecontest":
            by_id[prepared["hashed_id"]] = prepared
        prompt = prepared.get("task_prompt")
        if prompt:
            by_prompt[prompt] = prepared

    return by_id, by_prompt


def _sorted_items_from_payload(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, list):
        return [(str(index), item) for index, item in enumerate(payload) if isinstance(item, dict)]
    if isinstance(payload, dict):
        keys = list(payload.keys())
        try:
            sorted_keys = sorted(keys, key=lambda value: int(value) if str(value).isdigit() else str(value))
        except Exception:
            sorted_keys = sorted(keys, key=str)
        return [(str(key), payload[key]) for key in sorted_keys if isinstance(payload[key], dict)]
    raise TypeError(f"Unsupported JSON type: {type(payload)!r}")


def normalize_legacy_v1_code_record(item: dict[str, Any], fallback_id: str) -> dict[str, Any] | None:
    if "TaskId" not in item and "Attempt answer" not in item and "Raw response" not in item:
        return None

    phase = str(item.get("Phase", "")).strip().lower()
    if phase and phase != "eval":
        return None

    normalized = dict(item)
    record_id = item.get("TaskId", item.get("id", fallback_id))
    normalized["id"] = str(record_id)

    question = item.get("Question") or item.get("question")
    if isinstance(question, str) and question.strip():
        normalized["task_prompt"] = question

    attempt_answer = item.get("Attempt answer")
    if isinstance(attempt_answer, str) and attempt_answer.strip():
        normalized["hypothesis"] = attempt_answer

    raw_response = item.get("Raw response")
    if isinstance(raw_response, str):
        normalized["raw_response"] = raw_response

    if "Solved" in item:
        normalized["is_solved"] = item["Solved"]

    entry_point = item.get("Entry point")
    if isinstance(entry_point, str) and entry_point.strip():
        normalized["entry_point"] = entry_point

    return normalized


def is_legacy_v1_code_record(item: dict[str, Any]) -> bool:
    return "TaskId" in item or "Attempt answer" in item or "Raw response" in item


def load_result_records(path: Path) -> list[tuple[str, dict[str, Any]]]:
    if path.suffix == ".jsonl":
        records: list[tuple[str, dict[str, Any]]] = []
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    continue
                record_id = str(item.get("id", index))
                records.append((record_id, item))
        return records

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records: list[tuple[str, dict[str, Any]]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            normalized = normalize_legacy_v1_code_record(item, str(index))
            if normalized is not None:
                records.append((str(normalized["id"]), normalized))
                continue
            if is_legacy_v1_code_record(item):
                continue
            record_id = str(item.get("id", item.get("TaskId", index)))
            records.append((record_id, item))
        return records

    return _sorted_items_from_payload(payload)


def extract_code_from_response(response: Any) -> str:
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
        or "if __name__" in content
    ):
        return content

    return ""


def get_stored_solved_flag(record: dict[str, Any]) -> bool | None:
    for key in ("is_solved", "solved", "sovled", "Solved"):
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def run_with_timeout(func, timeout: int):
    result_container: list[Any] = []
    exception_container: list[BaseException] = []

    def target() -> None:
        try:
            result_container.append(func())
        except BaseException as exc:  # noqa: BLE001
            exception_container.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Execution timed out after {timeout} seconds")
    if exception_container:
        raise exception_container[0]
    return result_container[0] if result_container else None


class MBPPExecutor:
    def execute(self, func: str, test: str, timeout: int = 15) -> tuple[bool, str]:
        if not func:
            return False, "No code generated"
        if not test:
            return False, "No MBPP assertions provided"

        global_dict = {
            "math": __import__("math"),
            "re": __import__("re"),
            "sys": __import__("sys"),
            "os": __import__("os"),
            "random": __import__("random"),
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "heapq": __import__("heapq"),
            "typing": __import__("typing"),
        }
        try:
            import numpy

            global_dict["np"] = numpy
            global_dict["numpy"] = numpy
        except Exception:
            pass

        try:
            exec(func, global_dict)
            run_with_timeout(lambda: exec(test, global_dict), timeout)
            return True, "Tests passed"
        except TimeoutError as exc:
            return False, f"Timeout: {exc}"
        except Exception as exc:  # noqa: BLE001
            tb_list = traceback.format_tb(exc.__traceback__)
            relevant_tb = "".join(tb_list[-2:]) if tb_list else ""
            return False, f"{type(exc).__name__}: {exc}\n{relevant_tb}"


class HumanEvalExecutor:
    def execute(self, func: str, test: str, entry_point: str | None, timeout: int = 15) -> tuple[bool, str]:
        if not func:
            return False, "No code generated"
        if not test:
            return False, "No HumanEval test provided"

        global_dict = {
            "math": __import__("math"),
            "hashlib": __import__("hashlib"),
            "re": __import__("re"),
            "sys": __import__("sys"),
            "os": __import__("os"),
            "random": __import__("random"),
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "heapq": __import__("heapq"),
            "typing": __import__("typing"),
        }
        try:
            import numpy

            global_dict["np"] = numpy
            global_dict["numpy"] = numpy
        except Exception:
            pass

        code = func
        if entry_point == "decode_cyclic":
            code = (
                '\n\ndef encode_cyclic(s: str):\n'
                '    """\n'
                "    returns encoded string by cycling groups of three characters.\n"
                '    """\n'
                "    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n"
                "    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n"
                '    return "".join(groups)\n\n'
            ) + func
        elif entry_point == "decode_shift":
            code = (
                '\n\ndef encode_shift(s: str):\n'
                '    """\n'
                "    returns encoded string by shifting every character by 5 in the alphabet.\n"
                '    """\n'
                '    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n'
            ) + func
        elif entry_point == "find_zero":
            code = (
                "\n\ndef poly(xs: list, x: float):\n"
                "    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n"
            ) + func

        try:
            exec(code, global_dict)
            run_with_timeout(lambda: exec(test, global_dict), timeout)
            return True, "Tests passed"
        except TimeoutError as exc:
            return False, f"Timeout: {exc}"
        except Exception as exc:  # noqa: BLE001
            tb_list = traceback.format_tb(exc.__traceback__)
            relevant_tb = "".join(tb_list[-2:]) if tb_list else ""
            return False, f"{type(exc).__name__}: {exc}\n{relevant_tb}"


class CodeContestExecutor:
    def __init__(self, timeout: int = 2):
        self.timeout = timeout

    @staticmethod
    def normalize(output: str) -> str:
        if not output:
            return ""
        return "\n".join(line.rstrip() for line in output.strip().splitlines())

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

            solved = passed_count == total_count
            return solved, f"Passed {passed_count}/{total_count} cases", first_error
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
        return "\n".join(line.rstrip() for line in output.strip().splitlines())

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
                        input=str(case.get("input", "")).encode("utf-8"),
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
                    expected_out = str(case.get("output", ""))
                    if self.normalize(actual_out) == self.normalize(expected_out):
                        passed_count += 1
                    elif not first_error:
                        short_act = actual_out[:100] + "..." if len(actual_out) > 100 else actual_out
                        first_error = f"Wrong Answer on case {i}. Got: {short_act}"
                except subprocess.TimeoutExpired:
                    if not first_error:
                        first_error = f"TLE on case {i}"

            solved = passed_count == total_count
            return solved, f"Passed {passed_count}/{total_count} public cases", first_error
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


def evaluate_instance(task: str, code: str, instance: dict[str, Any]) -> tuple[bool, str, str]:
    if task == "mbpp":
        ok, feedback = MBPPExecutor().execute(code, instance["test"], timeout=15)
        return ok, feedback, ""
    if task == "humaneval":
        ok, feedback = HumanEvalExecutor().execute(code, instance["test"], instance.get("entry_point"), timeout=15)
        return ok, feedback, ""
    if task == "codecontest":
        return CodeContestExecutor(timeout=2).execute(code, instance["tests"])
    if task == "livecode":
        return LiveCodeExecutor(timeout=4).execute(code, instance["test_cases"])
    raise ValueError(f"Unsupported task: {task}")


def lookup_dataset_instance(
    task: str,
    record_id: str,
    record: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    by_prompt: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [record_id]

    if record.get("id") is not None:
        candidates.append(str(record["id"]))

    for candidate in candidates:
        if candidate in by_id:
            return by_id[candidate]

    prompt = (
        record.get("task_prompt")
        or record.get("prompt")
        or record.get("question")
        or record.get("question_content")
    )
    if isinstance(prompt, str) and prompt in by_prompt:
        return by_prompt[prompt]

    if task == "codecontest":
        task_prompt = record.get("prompt") or record.get("task_prompt")
        if isinstance(task_prompt, str):
            hashed_id = f"{record_id}_{_generate_short_hash(task_prompt)}" if str(record_id).isdigit() else None
            if hashed_id and hashed_id in by_id:
                return by_id[hashed_id]

    return None


def evaluate_result_file(
    task: str,
    result_file: Path,
    dataset_file: Path,
    trust_stored_flags: bool,
    limit: int | None,
) -> dict[str, Any]:
    records = load_result_records(result_file)
    if limit is not None:
        records = records[:limit]

    by_id, by_prompt = load_dataset_lookup(task, dataset_file)

    total_records = len(records)
    matched_records = 0
    correct = 0
    stored_total = 0
    stored_correct = 0
    changed_vs_stored = 0
    missing_dataset_records: list[str] = []
    mismatches: list[dict[str, Any]] = []

    for record_id, record in records:
        stored_flag = get_stored_solved_flag(record)
        if stored_flag is not None:
            stored_total += 1
            stored_correct += int(stored_flag)

        instance = lookup_dataset_instance(task, record_id, record, by_id, by_prompt)
        if instance is None:
            missing_dataset_records.append(record_id)
            continue

        matched_records += 1

        if trust_stored_flags:
            solved = bool(stored_flag)
            execution_result = record.get("execution_result", "stored_flag")
            error_text = str(record.get("error", ""))
        else:
            code = record.get("hypothesis")
            if not isinstance(code, str) or not code.strip():
                code = extract_code_from_response(record.get("raw_response", ""))
            solved, execution_result, error_text = evaluate_instance(task, code, instance)

        correct += int(solved)

        if stored_flag is not None and stored_flag != solved:
            changed_vs_stored += 1
            mismatches.append(
                {
                    "id": record_id,
                    "stored": stored_flag,
                    "rejudged": solved,
                    "execution_result": execution_result,
                    "error": error_text[:300],
                }
            )

    accuracy = round(correct / matched_records * 100, 4) if matched_records else 0.0
    stored_accuracy = round(stored_correct / stored_total * 100, 4) if stored_total else None
    paper_score = TASK_SPECS[task].paper_score

    return {
        "task": task,
        "paper_name": TASK_SPECS[task].paper_name,
        "judge_note": TASK_SPECS[task].judge_note,
        "result_file": str(result_file),
        "dataset_file": str(dataset_file),
        "paper_score": paper_score,
        "paper_setting": "simple fixed indicators (w/ Generic Indicators)",
        "total_records": total_records,
        "matched_records": matched_records,
        "missing_dataset_records": missing_dataset_records,
        "correct": correct,
        "accuracy": accuracy,
        "delta_vs_paper": round(accuracy - paper_score, 4),
        "stored_total": stored_total,
        "stored_correct": stored_correct,
        "stored_accuracy": stored_accuracy,
        "changed_vs_stored": changed_vs_stored,
        "trust_stored_flags": trust_stored_flags,
        "mismatches": mismatches,
    }


def accuracy_fraction(correct: int, total: int) -> float | None:
    if total <= 0:
        return None
    return correct / total


def align_summary_with_paper_code_eval(
    task: str,
    summary_path: Path,
    result_file: Path,
    dataset_file: str | Path | None = None,
) -> dict[str, Any]:
    canonical_task = ensure_task_name(task)
    dataset_arg: str | None = None
    if dataset_file:
        candidate = Path(dataset_file)
        if candidate.exists():
            dataset_arg = str(candidate)

    resolved_dataset = resolve_dataset_path(canonical_task, dataset_arg)
    paper_summary = evaluate_result_file(
        canonical_task,
        result_file,
        resolved_dataset,
        trust_stored_flags=False,
        limit=None,
    )

    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                summary = payload
        except Exception:
            summary = {}

    eval_section = summary.get("eval")
    if not isinstance(eval_section, dict):
        eval_section = {}
        summary["eval"] = eval_section

    eval_section["solved"] = paper_summary["correct"]
    eval_section["executed"] = paper_summary["matched_records"]
    if not eval_section.get("num_samples"):
        eval_section["num_samples"] = paper_summary["matched_records"]
    eval_section["accuracy"] = accuracy_fraction(paper_summary["correct"], paper_summary["matched_records"])

    summary["task"] = str(summary.get("task") or canonical_task)
    summary["domain"] = str(summary.get("domain") or canonical_task)
    summary["result_file"] = str(result_file)
    summary["summary_file"] = str(summary_path)
    summary["dataset_json"] = str(summary.get("dataset_json") or resolved_dataset)
    summary["paper_code_eval"] = paper_summary
    summary["paper_setting"] = paper_summary["paper_setting"]
    summary["judge_note"] = paper_summary["judge_note"]
    summary["summary_aligned_with"] = "paper_code_eval"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def print_reference_table() -> None:
    print("task,paper_name,paper_score,paper_setting,judge_note")
    total = 0.0
    for task in ("mbpp", "humaneval", "codecontest", "livecode"):
        spec = TASK_SPECS[task]
        print(
            f"{task},{spec.paper_name},{spec.paper_score:.2f},"
            "simple fixed indicators (w/ Generic Indicators),"
            f"{spec.judge_note}"
        )
        total += spec.paper_score
    print(f"avg,Average,{total / 4:.2f},simple fixed indicators (w/ Generic Indicators),Table 3 reference")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--dataset-file", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trust-stored-flags", action="store_true")
    parser.add_argument("--show-reference-table", action="store_true")
    parser.add_argument("--show-mismatches", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.show_reference_table:
        print_reference_table()
        return 0

    if not args.task or not args.result_file:
        parser.error("--task and --result-file are required unless --show-reference-table is used")

    task = ensure_task_name(args.task)
    result_file = Path(args.result_file)
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    dataset_file = resolve_dataset_path(task, args.dataset_file)
    summary = evaluate_result_file(task, result_file, dataset_file, args.trust_stored_flags, args.limit)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    print(f"task               : {summary['paper_name']} ({summary['task']})")
    print(f"paper_setting      : {summary['paper_setting']}")
    print(f"judge_note         : {summary['judge_note']}")
    print(f"result_file        : {summary['result_file']}")
    print(f"dataset_file       : {summary['dataset_file']}")
    print(f"paper_score        : {summary['paper_score']:.2f}")
    print(f"matched_records    : {summary['matched_records']}/{summary['total_records']}")
    print(f"correct            : {summary['correct']}")
    print(f"accuracy           : {summary['accuracy']:.4f}")
    print(f"delta_vs_paper     : {summary['delta_vs_paper']:+.4f}")
    if summary["stored_total"]:
        print(f"stored_accuracy    : {summary['stored_accuracy']:.4f}")
        print(f"changed_vs_stored  : {summary['changed_vs_stored']}")
    else:
        print("stored_accuracy    : n/a")

    if summary["missing_dataset_records"]:
        print(f"missing_dataset    : {len(summary['missing_dataset_records'])}")
        for record_id in summary["missing_dataset_records"][: args.show_mismatches]:
            print(f"  - {record_id}")

    if summary["mismatches"]:
        print("mismatches         :")
        for item in summary["mismatches"][: args.show_mismatches]:
            print(
                f"  - {item['id']}: stored={item['stored']} rejudged={item['rejudged']} "
                f"result={item['execution_result']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
