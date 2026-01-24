from abc import ABC, abstractmethod
from collections.abc import Sequence

from .context import AnyChatCompletionMessage
from .tool import ToolAny


class ContextRefiner(ABC):
    @abstractmethod
    async def refine_new_messages(
        self, raw_messages: Sequence[AnyChatCompletionMessage], refined_messages: Sequence[AnyChatCompletionMessage]
    ) -> Sequence[AnyChatCompletionMessage]:
        pass

    @abstractmethod
    def get_injected_tools(self) -> list[ToolAny]:
        pass
