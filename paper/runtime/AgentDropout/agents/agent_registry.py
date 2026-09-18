from typing import Type
from class_registry import ClassRegistry

from autogen_agentchat.agents import BaseChatAgent


class AgentRegistry:
    registry = ClassRegistry(attr_name="agent_name")

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, agent_name: str, *args, **kwargs) -> type[BaseChatAgent]:
        return cls.registry.get(agent_name, *args, **kwargs)

    @classmethod
    def get_class(cls, agent_name: str) -> Type:
        return cls.registry.get_class(agent_name)
