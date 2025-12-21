from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from easy_agents.utils.types.json_dict import JsonType


class AbstractTool(ABC):
    """
    Base class for all tools. Each tool must define a Parameters Pydantic model
    and implement a `run` method.
    """

    @classmethod
    @abstractmethod
    def get_params_type(cls) -> type[BaseModel]:
        raise NotImplementedError

    @classmethod
    def run(cls, **kwargs: Any) -> str:
        parameters = cls.get_params_type()(**kwargs)
        return cls._run(parameters)

    @classmethod
    @abstractmethod
    def _run(cls, params: BaseModel) -> str:
        """Run the tool with validated parameters."""
        pass

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def schema(cls) -> JsonType:
        """Return the JSON schema of the tool parameters for AI consumption."""
        return {
            "name": cls.name(),
            "description": cls.description(),
            "parameters": cls.get_params_type().model_json_schema(),
        }
