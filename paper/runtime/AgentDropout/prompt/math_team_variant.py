from __future__ import annotations

import os


_ENABLED_VARIANTS = {"verifier", "lean_verifier", "math_team_v2"}


def variant_enabled() -> bool:
    variant = os.environ.get("AGENTDROPOUT_MATH_TEAM_VARIANT", "").strip().lower()
    return variant in _ENABLED_VARIANTS


def apply_math_team_variant(
    role_description: dict[str, str],
    description: dict[str, str],
    few_shot_data: dict[str, str],
    *,
    task_label: str,
    final_answer_instruction: str,
    answer_example: str | None = None,
) -> None:
    if not variant_enabled():
        return

    role_description["Programming Expert"] = (
        f"You are a computational verifier for {task_label}. "
        "You will be given a math problem together with drafts from other agents. "
        "Your job is to check arithmetic, small-case enumeration, algebraic identities, edge cases, and final-answer consistency. "
        "Use the smallest useful verification step. "
        "If a short Python snippet helps, keep it minimal and tightly targeted. "
        "Do not write long boilerplate programs, generic tutorials, or a full re-solution unless the earlier drafts are unusable. "
        "If code is unnecessary, write a concise verification note in natural language. "
        f"{final_answer_instruction}"
    )
    description["Programming Expert"] = (
        "A verification-oriented agent that checks decisive steps, runs short computations when useful, "
        "and avoids long programming detours."
    )
    few_shot_data["Programming Expert"] = (
        "Q: Verify whether the claimed sum of the integers from 1 to 10 equals 55.\n"
        "A:\n"
        "Check the closed form: 1 + 2 + \\cdots + 10 = 10 \\cdot 11 / 2 = 55, so the claim is correct.\n"
        "If this check maps directly to the target task, report the verified result in the required final format.\n\n"
        "Q: Verify how many integers x in [1, 20] satisfy x^2 < 50.\n"
        "A:\n"
        "```python\n"
        "valid = [x for x in range(1, 21) if x * x < 50]\n"
        "answer = len(valid)\n"
        "```\n"
        "The short check finds 7 valid integers.\n"
        "Use that evidence to support the final answer in the required format."
    )
    if "Inspector" in role_description:
        role_description["Inspector"] += (
            " Focus on the smallest decisive flaw or confirmation. "
            "Do not restate the whole solution when a short audit is enough."
        )
