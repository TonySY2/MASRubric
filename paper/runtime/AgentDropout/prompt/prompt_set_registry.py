from typing import Type
from class_registry import ClassRegistry

from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.teamv2_math_prompt_variants import load_prompt_variants, resolve_prompt_name

class PromptSetRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        load_prompt_variants()
        return cls.registry.keys()

    @classmethod
    def get(cls, name: str, *args, **kwargs) -> PromptSet:
        load_prompt_variants()
        resolved_name = resolve_prompt_name(name)
        return cls.registry.get(resolved_name, *args, **kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type:
        load_prompt_variants()
        resolved_name = resolve_prompt_name(name)
        return cls.registry.get_class(resolved_name)
