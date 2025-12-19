from abc import ABC, abstractmethod


class BaseTool(ABC):
    @property
    @abstractmethod
    def tool_description(self) -> str:
        """AI-readable description of the tool.

        should be the following format:
        tool name: Tool description.
            - tool_name: <tool name>
            - tool_parameters: A json object with the following format: ...
            - extra things about the tool if needed.
        """
        raise NotImplementedError
