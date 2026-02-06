from typing import Any

import jinja2
from pydantic import BaseModel, Field, PrivateAttr

from easy_agents.core.agent import AgentInputType
from easy_agents.core.context import AnyChatCompletionMessage, Context


class ContextFactory[InputT: AgentInputType, DepsT: Any](BaseModel):
    messages_templates: list[AnyChatCompletionMessage] = Field(default_factory=list)

    _compiled_templates: list[jinja2.Template] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Compile templates once when the factory is created."""
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self._compiled_templates = [env.from_string(str(m.content)) for m in self.messages_templates]

    def __call__(self, input_args: InputT, deps: DepsT) -> Context:
        rendered: list[AnyChatCompletionMessage] = []

        for template, original in zip(self._compiled_templates, self.messages_templates, strict=True):
            content = template.render(input=input_args, deps=deps)
            rendered.append(original.model_copy(update={"content": content}))

        return Context(messages=rendered)
