from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"

DEFAULT_REASONING_MODEL = "Qwen/Qwen3-8B"
DEFAULT_SUPERVISOR_MODEL = "Qwen/Qwen3-8B"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_SELECTOR_MODEL = os.environ.get("MASRUBRIC_V2_SELECTOR_MODEL", "Qwen/Qwen3.5-9B")
DEFAULT_SELECTOR_URL = os.environ.get(
    "MASRUBRIC_V2_SELECTOR_URL",
    "http://localhost:8001/v1",
)
DEFAULT_SELECTOR_API_KEY = os.environ.get(
    "MASRUBRIC_V2_SELECTOR_API_KEY",
    "EMPTY",
)
DEFAULT_PYTHON_BIN = os.environ.get(
    "MASRUBRIC_V2_PYTHON_BIN",
    sys.executable,
)

DEFAULT_CODE_METRIC_POOL_FILE = str(ROOT.parent / "assets/pools/code/deduplicated_metrics_pool.json")
DEFAULT_CODE_EMBEDDING_CACHE_FILE = str(ROOT.parent / "assets/pools/code/deduplicated_embeddings-trigger.jsonl")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain_dir: str
    agent_name: str
    domain_name: str
    decision_agent_name: str
    dataset_path: str
    dataset_format: str


TASK_ORDER = [
    "mbpp",
    "humaneval",
    "codecontest",
    "livecode",
    "apps_competition",
]


TASK_SPECS = {
    "mbpp": TaskSpec(
        name="mbpp",
        domain_dir="mbpp",
        agent_name="CodeWriting_mbpp",
        domain_name="mbpp",
        decision_agent_name="FinalWriteCodeMBPP",
        dataset_path=str(ROOT.parent / "assets/datasets/mbpp/mbpp_test.jsonl"),
        dataset_format="mbpp",
    ),
    "humaneval": TaskSpec(
        name="humaneval",
        domain_dir="humaneval",
        agent_name="CodeWriting_humaneval",
        domain_name="humaneval",
        decision_agent_name="FinalWriteCode",
        dataset_path=str(ROOT.parent / "assets/datasets/humaneval/humaneval-py.jsonl"),
        dataset_format="humaneval",
    ),
    "codecontest": TaskSpec(
        name="codecontest",
        domain_dir="codecontest",
        agent_name="CodeWriting_codecontest",
        domain_name="codecontest",
        decision_agent_name="FinalWriteCode",
        dataset_path=str(ROOT.parent / "assets/datasets/codecontest/test.jsonl"),
        dataset_format="codecontest",
    ),
    "livecode": TaskSpec(
        name="livecode",
        domain_dir="livecode",
        agent_name="CodeWriting_livecode",
        domain_name="livecode",
        decision_agent_name="FinalWriteCode",
        dataset_path=str(ROOT.parent / "assets/datasets/livecode/livecodebench_v1.jsonl"),
        dataset_format="livecode",
    ),
    "apps_competition": TaskSpec(
        name="apps_competition",
        domain_dir="apps_competition",
        agent_name="CodeWriting_codecontest",
        domain_name="codecontest",
        decision_agent_name="FinalWriteCode",
        dataset_path=str(ROOT.parent / "assets/datasets/apps/apps_competition_test.jsonl"),
        dataset_format="apps",
    ),
    "taco": TaskSpec(
        name="taco",
        domain_dir="taco",
        agent_name="CodeWriting_codecontest",
        domain_name="codecontest",
        decision_agent_name="FinalWriteCode",
        dataset_path=str(ROOT.parent / "assets/datasets/taco/taco_test.jsonl"),
        dataset_format="apps",
    ),
}


TASK_ALIASES = {
    "human_eval": "humaneval",
    "human-eval": "humaneval",
    "codecontests": "codecontest",
    "code_contests": "codecontest",
    "code-contests": "codecontest",
    "livecodebench": "livecode",
    "live_codebench": "livecode",
    "live_code_bench": "livecode",
    "live-code-bench": "livecode",
    "livecode_bench": "livecode",
    "apps": "apps_competition",
    "apps_competition": "apps_competition",
    "apps-competition": "apps_competition",
    "appscompetition": "apps_competition",
    "taco": "taco",
    "baai_taco": "taco",
    "baai-taco": "taco",
}


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def ensure_task_name(task: str) -> TaskSpec:
    task = task.strip()
    if task not in TASK_SPECS:
        normalized = task.casefold().replace("-", "_").replace(" ", "_")
        canonical = TASK_ALIASES.get(normalized)
        if canonical is None:
            for existing in TASK_SPECS:
                if existing.casefold() == task.casefold():
                    canonical = existing
                    break
        if canonical is None:
            raise KeyError(f"Unknown task: {task}")
        task = canonical
    return TASK_SPECS[task]
