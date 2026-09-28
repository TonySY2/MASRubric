from __future__ import annotations

import importlib
import os
from typing import Iterable


TEAMV2_ENV = "MASRUBRIC_MATH_PROMPT_VARIANT"

_VARIANT_MODULES = {
    "teamv2": [
        "masrubric.prompt.math_prompt_set_teamv2",
        "masrubric.prompt.aqua_prompt_set_teamv2",
        "masrubric.prompt.gsm8k_prompt_set_teamv2",
        "masrubric.prompt.math500_prompt_set_teamv2",
        "masrubric.prompt.amc23_prompt_set_teamv2",
        "masrubric.prompt.aime24_prompt_set_teamv2",
        "masrubric.prompt.aime25_prompt_set_teamv2",
        "masrubric.prompt.olympiad_prompt_set_teamv2",
        "masrubric.prompt.olymMATH_prompt_set_teamv2",
    ],
}

_DOMAIN_VARIANT_KEYS = {
    "teamv2": {
        "math_nocot": "math_nocot_teamv2",
        "aqua": "aqua_teamv2",
        "gsm8k": "gsm8k_teamv2",
        "math500": "math500_teamv2",
        "amc23": "amc23_teamv2",
        "aime24": "aime24_teamv2",
        "aime25": "aime25_teamv2",
        "olympiad": "olympiad_teamv2",
        "olymmath": "olymMATH_teamv2",
    },
}


def normalize_variant_name(raw_variant: str | None) -> str:
    if not raw_variant:
        return ""
    normalized = raw_variant.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"base", "default", "original", "legacy", "off", "none"}:
        return ""
    return normalized


def get_active_variant() -> str:
    return normalize_variant_name(os.environ.get(TEAMV2_ENV))


def resolve_prompt_name(domain: str) -> str:
    variant = get_active_variant()
    if not variant:
        return domain

    domain_key = domain.strip().casefold()
    return _DOMAIN_VARIANT_KEYS.get(variant, {}).get(domain_key, domain)


def load_prompt_variants() -> None:
    variant = get_active_variant()
    if not variant:
        return

    module_names = _VARIANT_MODULES.get(variant, [])
    for module_name in module_names:
        importlib.import_module(module_name)

