from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Any

import httpx
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core.models import RequestUsage
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)

from masrubric.usage_tracking import record_chat_completion
from project_datasets.aqua_dataset import aqua_get_predict


AGENT_MODE_ALIASES = {
    "supervisor": "supervisor",
    "v2": "supervisor",
    "audit": "supervisor",
    "baseline": "supervisor",
    "prm": "prm",
    "best_of_n": "prm",
    "best-of-n": "prm",
    "lets_verify_step_by_step": "prm",
    "let_s_verify_step_by_step": "prm",
    "self_refine": "self_refine",
    "self-refine": "self_refine",
    "selfrefine": "self_refine",
    "multi_tag": "multi_tag",
    "multi-tag": "multi_tag",
    "multitag": "multi_tag",
}


def normalize_agent_mode(raw_mode: Any) -> str:
    text = str(raw_mode or "supervisor").strip()
    key = text.casefold().replace(" ", "_")
    mode = AGENT_MODE_ALIASES.get(text, AGENT_MODE_ALIASES.get(key))
    if mode is None:
        raise ValueError(f"Unsupported agent mode: {raw_mode}")
    return mode


def agent_mode_enabled(agent: Any) -> bool:
    return normalize_agent_mode(getattr(agent, "agent_mode", "supervisor")) != "supervisor"


def configure_agent_mode(
    agent: Any,
    *,
    agent_mode: Any = "supervisor",
    prm_url: str | None = None,
    prm_n_samples: int = 3,
    shared_score_board: list[BaseChatMessage] | None = None,
    predictor: "PredictorAgent | None" = None,
) -> None:
    mode = normalize_agent_mode(agent_mode)
    agent.agent_mode = mode
    agent.prm_url = prm_url
    agent.prm_n_samples = int(prm_n_samples or 1)
    agent._mode_time = 0.0
    agent.shared_score_board = shared_score_board if shared_score_board is not None else []
    agent.predictor = predictor


def _agent_task_kind(agent: Any) -> str:
    class_name = agent.__class__.__name__.casefold()
    domain = str(getattr(getattr(agent, "prompt_set", None), "domain", "") or "").casefold()
    if "codewriting" in class_name or domain in {"mbpp", "humaneval", "codecontest", "livecode"}:
        return "code"
    return "math"


async def _build_agent_context(agent: Any, question: str) -> list[dict[str, str]]:
    custom_builder = getattr(agent, "_build_agent_mode_context", None)
    if callable(custom_builder):
        return await custom_builder(question, agent._message_history)
    system_prompt, user_prompt = await agent._process_inputs(task=question, input_messages=agent._message_history)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class PredictorAgent:
    def __init__(self, model: str, api_key: str, base_url: str, task_kind: str = "math"):
        self.model = model
        self.task_kind = task_kind
        self._model_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )
        self.is_completed = False
        self._answer_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def predict_and_record(self, question: str, solver_output: str) -> str | None:
        if self.is_completed:
            return None

        if self.task_kind == "code":
            prompt = (
                "You are evaluating a programming problem and a candidate code solution. "
                "Judge whether the candidate is likely to satisfy the task, including algorithmic correctness, "
                "edge cases, complexity, and required input/output or function format.\n\n"
                f"Problem:\n{question}\n\n"
                f"Candidate Solution:\n{solver_output}\n\n"
                "Return a brief assessment and end with exactly one line: TAG: correct, TAG: incorrect, "
                "or TAG: uncertain."
            )
        else:
            prompt = (
                "You are evaluating a math problem and a proposed solution. "
                "Carefully reason about whether the solution is correct and what the final answer should be.\n\n"
                f"Question:\n{question}\n\n"
                f"Proposed Solution:\n{solver_output}\n\n"
                "Based on your analysis, determine the final answer. "
                "You may reason freely, but you MUST end your response by writing the final answer "
                r"in the format: \boxed{<answer>} where <answer> is the final result."
            )

        try:
            completion = await self._model_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            record_chat_completion(
                completion,
                model=self.model,
                stage="agent_multi_tag_predictor",
                source="PredictorAgent",
            )
            predicted_text = completion.choices[0].message.content or ""
            predicted_answer = _extract_prediction(predicted_text, task_kind=self.task_kind)
        except Exception as exc:
            logging.warning(f"[PredictorAgent] LLM call failed: {exc}")
            return None

        if predicted_answer:
            async with self._lock:
                self._answer_counts[predicted_answer] = self._answer_counts.get(predicted_answer, 0) + 1
                self._check_completion()
                print(
                    f"[PredictorAgent] prediction='{predicted_answer}' "
                    f"counts={self._answer_counts} completed={self.is_completed}"
                )
        else:
            print(f"[PredictorAgent] failed to extract answer from: {predicted_text}")

        return predicted_answer

    def _check_completion(self) -> None:
        if not self._answer_counts:
            return
        if self.task_kind == "code":
            if self._answer_counts.get("correct", 0) >= 2:
                self.is_completed = True
            return
        sorted_counts = sorted(self._answer_counts.values(), reverse=True)
        top = sorted_counts[0]
        second = sorted_counts[1] if len(sorted_counts) > 1 else 0
        if top - second >= 2:
            self.is_completed = True

    def reset(self) -> None:
        self.is_completed = False
        self._answer_counts = {}


def configure_agent_mode_team(
    participants: Sequence[Any],
    *,
    agent_mode: Any,
    prm_url: str | None,
    prm_n_samples: int,
    model: str,
    api_key: str,
    base_url: str,
) -> tuple[list[BaseChatMessage], PredictorAgent | None]:
    shared_score_board: list[BaseChatMessage] = []
    mode = normalize_agent_mode(agent_mode)
    task_kind = _agent_task_kind(participants[0]) if participants else "math"
    predictor = (
        PredictorAgent(model=model, api_key=api_key, base_url=base_url, task_kind=task_kind)
        if mode == "multi_tag"
        else None
    )
    for agent in participants:
        configure_agent_mode(
            agent,
            agent_mode=mode,
            prm_url=prm_url,
            prm_n_samples=prm_n_samples,
            shared_score_board=shared_score_board,
            predictor=predictor,
        )
    return shared_score_board, predictor


def attach_agent_mode_state(
    team: Any,
    shared_score_board: list[BaseChatMessage],
    predictor: PredictorAgent | None,
) -> None:
    team.agent_mode_shared_score_board = shared_score_board
    team.agent_mode_predictor = predictor


def reset_agent_mode_state(team: Any) -> None:
    shared_score_board = getattr(team, "agent_mode_shared_score_board", None)
    if isinstance(shared_score_board, list):
        shared_score_board.clear()
    predictor = getattr(team, "agent_mode_predictor", None)
    if predictor is not None and hasattr(predictor, "reset"):
        predictor.reset()
    for participant in getattr(team, "_participants", []) or []:
        if hasattr(participant, "_mode_time"):
            participant._mode_time = 0.0


def get_agent_mode_history(team: Any) -> list[BaseChatMessage]:
    shared_score_board = getattr(team, "agent_mode_shared_score_board", [])
    return [
        msg
        for msg in shared_score_board
        if getattr(msg, "source", None) != "user" and getattr(msg, "content", "") != "[skipped]"
    ]


async def emit_agent_mode_response(agent: Any, question: str, *, max_tokens: int) -> Response:
    mode = normalize_agent_mode(getattr(agent, "agent_mode", "supervisor"))
    if mode == "prm":
        response_dict, completion = await _run_prm_mode(agent, question, max_tokens=max_tokens)
    elif mode == "self_refine":
        response_dict, completion = await _run_self_refine_mode(agent, question, max_tokens=max_tokens)
    elif mode == "multi_tag":
        response_dict, completion = await _run_multi_tag_mode(agent, question, max_tokens=max_tokens)
    else:
        raise ValueError(f"agent_mode helper received supervisor mode for {agent.name}")

    usage = _request_usage_from_completion(completion)
    response_message = TextMessage(
        content=response_dict.get("content", ""),
        source=agent.name,
        models_usage=usage,
    )
    shared_score_board = getattr(agent, "shared_score_board", None)
    if isinstance(shared_score_board, list):
        shared_score_board.append(response_message)
    agent._message_history.append(response_message)
    return Response(chat_message=response_message)


async def _run_prm_mode(agent: Any, question: str, *, max_tokens: int) -> tuple[dict[str, Any], Any | None]:
    prm_url = str(getattr(agent, "prm_url", "") or "").strip()
    task_kind = _agent_task_kind(agent)
    conversation = await _build_agent_context(agent, question)

    async def one_sample(index: int) -> tuple[dict[str, Any], Any]:
        completion = await _chat_completion(
            agent,
            messages=conversation,
            max_tokens=max_tokens,
            stage="agent_prm_candidate",
            metadata={"candidate_index": index, "prm_n_samples": int(getattr(agent, "prm_n_samples", 1))},
        )
        return completion.choices[0].message.model_dump(), completion

    sample_count = max(1, int(getattr(agent, "prm_n_samples", 1) or 1))
    t0 = asyncio.get_running_loop().time()
    pairs = await asyncio.gather(*[one_sample(idx) for idx in range(sample_count)])
    agent._mode_time = getattr(agent, "_mode_time", 0.0) + asyncio.get_running_loop().time() - t0

    if _uses_external_prm(prm_url):
        scores = await asyncio.gather(*[_call_prm_api(agent, question, pair[0].get("content", "")) for pair in pairs])
        best_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
        print(
            f"[Best-of-{sample_count}] Agent '{agent.name}' PRM scores: "
            f"{[f'{score:.4f}' for score in scores]} -> best=#{best_idx} ({scores[best_idx]:.4f})"
        )
    else:
        best_idx, judge_text = await _choose_prm_candidate_with_llm(
            agent,
            question,
            [pair[0].get("content", "") for pair in pairs],
            task_kind=task_kind,
        )
        print(f"[Best-of-{sample_count}] Agent '{agent.name}' LLM judge -> best=#{best_idx}; judge={judge_text[:300]}")
    return pairs[best_idx][0], pairs[best_idx][1]


async def _call_prm_api(agent: Any, question: str, content: str) -> float:
    payload = {
        "system": getattr(agent, "_system_message", ""),
        "query": question,
        "response": [content],
    }
    url = str(getattr(agent, "prm_url", "") or "").rstrip("/") + "/v1/step_rewards"
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=1200.0, trust_env=False) as client:
        for attempt in range(4):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                retryable_status = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {429, 500, 502, 503, 504}
                if not retryable_status and not isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                    raise
                if attempt == 3:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"PRM request failed without response: {last_error}")
    scores = response.json().get("step_rewards", [])
    if not scores:
        return 0.0
    return float(min(scores))


def _uses_external_prm(prm_url: str | None) -> bool:
    text = str(prm_url or "").strip().casefold()
    return bool(text) and text not in {"self", "local", "reasoning", "reasoning_model", "llm", "llm_judge"}


async def _choose_prm_candidate_with_llm(
    agent: Any,
    question: str,
    candidates: list[str],
    *,
    task_kind: str = "math",
) -> tuple[int, str]:
    candidate_blocks = "\n\n".join(
        f"Candidate {idx}:\n{content}" for idx, content in enumerate(candidates)
    )
    if task_kind == "code":
        judge_prompt = (
            "You are a careful programming-solution evaluator. Select the best candidate code solution "
            "for the problem. Prefer candidates that satisfy the required function or stdin/stdout format, "
            "handle edge cases, and use an appropriate algorithmic complexity. "
            "Return a compact JSON object with keys best_index and reason.\n\n"
            f"Problem:\n{question}\n\n"
            f"{candidate_blocks}\n\n"
            "Use zero-based indexing for best_index."
        )
        system_content = "You judge programming solution candidates."
    else:
        judge_prompt = (
            "You are a careful math solution evaluator. Select the best candidate solution for the problem. "
            "Prefer the candidate with correct reasoning and a correct final answer. "
            "Return a compact JSON object with keys best_index and reason.\n\n"
            f"Problem:\n{question}\n\n"
            f"{candidate_blocks}\n\n"
            "Use zero-based indexing for best_index."
        )
        system_content = "You judge math solution candidates."
    completion = await _chat_completion(
        agent,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": judge_prompt},
        ],
        max_tokens=512,
        stage="agent_prm_llm_judge",
        metadata={"candidate_count": len(candidates)},
    )
    text = completion.choices[0].message.content or ""
    match = re.search(r'"best_index"\s*:\s*(\d+)', text)
    if not match:
        match = re.search(r"\bbest[_ -]?index\b\D{0,20}(\d+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(?:candidate|index|#)\s*(\d+)\b", text, flags=re.IGNORECASE)
    best_idx = int(match.group(1)) if match else 0
    if best_idx < 0 or best_idx >= len(candidates):
        best_idx = 0
    return best_idx, text


async def _run_self_refine_mode(agent: Any, question: str, *, max_tokens: int) -> tuple[dict[str, Any], Any | None]:
    task_kind = _agent_task_kind(agent)
    base_context = await _build_agent_context(agent, question)

    first_completion = await _chat_completion(
        agent,
        messages=base_context,
        max_tokens=max_tokens,
        stage="agent_self_refine_initial",
    )
    first_response = first_completion.choices[0].message.model_dump()
    first_content = first_response.get("content", "")
    _print_agent_block(agent.name, "[Round 1] Initial Answer", first_content)

    if task_kind == "code":
        critique_request = (
            "Please critically review your previous code solution. Identify algorithmic mistakes, "
            "edge cases, complexity problems, and function or input/output format issues. "
            "Be specific about what should be corrected."
        )
    else:
        critique_request = (
            "Please critically review your previous answer. "
            "Identify any logical errors, calculation mistakes, or areas that need improvement. "
            "Be specific about what should be corrected."
        )
    critique_conversation = base_context + [
        {"role": "assistant", "content": first_content},
        {"role": "user", "content": critique_request},
    ]
    t0 = asyncio.get_running_loop().time()
    critique_completion = await _chat_completion(
        agent,
        messages=critique_conversation,
        max_tokens=1024,
        stage="agent_self_refine_critique",
    )
    agent._mode_time = getattr(agent, "_mode_time", 0.0) + asyncio.get_running_loop().time() - t0
    critique_content = critique_completion.choices[0].message.content or ""
    _print_agent_block(agent.name, "[Reflection] Self-Critique", critique_content)

    if task_kind == "code":
        refine_request = (
            "Based on your self-critique above, provide a corrected and improved final solution. "
            "Include the complete Python code in a code block and address all issues you identified."
        )
    else:
        refine_request = (
            "Based on your self-critique above, please provide a corrected and improved answer. "
            "Make sure to address all the issues you identified."
        )
    refine_conversation = critique_conversation + [
        {"role": "assistant", "content": critique_content},
        {"role": "user", "content": refine_request},
    ]
    final_completion = await _chat_completion(
        agent,
        messages=refine_conversation,
        max_tokens=max_tokens,
        stage="agent_self_refine_refined",
    )
    final_response = final_completion.choices[0].message.model_dump()
    _print_agent_block(agent.name, "[Round 2] Refined Answer", final_response.get("content", ""))
    return final_response, final_completion


async def _run_multi_tag_mode(agent: Any, question: str, *, max_tokens: int) -> tuple[dict[str, Any], Any | None]:
    predictor = getattr(agent, "predictor", None)
    if predictor is not None and getattr(predictor, "is_completed", False):
        print(f"[multi-tag] Agent '{agent.name}' skipping because predictor completed.")
        return {"content": "[skipped]"}, None

    conversation = await _build_agent_context(agent, question)
    t0 = asyncio.get_running_loop().time()
    completion = await _chat_completion(
        agent,
        messages=conversation,
        max_tokens=max_tokens,
        stage="agent_multi_tag_solver",
    )
    response_dict = completion.choices[0].message.model_dump()
    content = response_dict.get("content", "")
    _print_agent_block(agent.name, "[multi-tag] Generated Content", content)
    if predictor is not None:
        await predictor.predict_and_record(question, content)
    agent._mode_time = getattr(agent, "_mode_time", 0.0) + asyncio.get_running_loop().time() - t0
    return response_dict, completion


async def _chat_completion(
    agent: Any,
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    try:
        completion = await agent._model_client.chat.completions.create(
            model=agent.model,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except (APITimeoutError, APIConnectionError, APIStatusError, httpx.ConnectTimeout) as exc:
        logging.warning(f"Agent '{agent.name}' API request failed in {stage}: {exc}")
        raise
    record_chat_completion(completion, model=agent.model, stage=stage, source=agent.name, metadata=metadata)
    return completion


def _request_usage_from_completion(completion: Any | None) -> RequestUsage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)
    return RequestUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _extract_prediction(text: str, *, task_kind: str = "math") -> str | None:
    if task_kind == "code":
        match = re.search(r"\bTAG\s*:\s*(correct|incorrect|uncertain)\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).casefold()
        lowered = text.casefold()
        if "likely correct" in lowered or "appears correct" in lowered:
            return "correct"
        if "incorrect" in lowered or "bug" in lowered or "fails" in lowered:
            return "incorrect"
        return "uncertain"
    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed
    predicted = aqua_get_predict(text)
    if predicted and predicted != "None.":
        return predicted
    return None


def _extract_last_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if not matches:
        return None
    return matches[-1].strip()


def _print_agent_block(agent_name: str, title: str, content: str) -> None:
    print(f"\n{'-' * 30}")
    print(f"{title} Agent '{agent_name}':")
    print(f"{'-' * 30}")
    print(content)
    print(f"{'-' * 50}\n")
