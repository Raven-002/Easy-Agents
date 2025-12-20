from abc import ABC, abstractmethod

from pydantic import BaseModel


class AbstractTool(ABC):
    """
    Base class for all tools. Each tool must define a Parameters Pydantic model
    and implement a `run` method.
    """

    class Parameters(BaseModel):
        """Define the parameters schema in subclasses."""

        place_holder: str

    @classmethod
    def run(cls, **kwargs) -> str:
        parameters = cls.Parameters(**kwargs)
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
    def schema(cls):
        """Return the JSON schema of the tool parameters for AI consumption."""
        return {"name": cls.name(), "description": cls.description(), "parameters": cls.Parameters.model_json_schema()}
