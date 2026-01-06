from collections.abc import Iterable
from dataclasses import dataclass

import openai
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionToolChoiceOptionParam
from openai.types.shared_params import ResponseFormatJSONSchema, ResponseFormatText
from openai.types.shared_params.response_format_json_schema import JSONSchema
from pydantic import BaseModel

from .context import AssistantMessage, ChatCompletionMessage
from .tool import BaseTool


@dataclass
class DescribedBaseModel:
    name: str
    description: str
    model: type[BaseModel]


class Model(BaseModel):
    api_base: str
    api_key: str
    model_name: str

    def create_openai_client(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.api_key, base_url=self.api_base)

    def chat_completion(
        self,
        messages: Iterable[ChatCompletionMessage],
        tools: list[BaseTool] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam = "auto",
        response_format: DescribedBaseModel | None = None,
        assistant_name: str = "",
        temperature: float = 0.0,
    ) -> AssistantMessage:
        tools_param = [t.get_json_schema() for t in tools] if tools else []

        response_format_param = (
            ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(
                    name=response_format.name,
                    description=response_format.description,
                    schema=response_format.model.model_json_schema(),
                    strict=True,
                ),
            )
            if response_format
            else ResponseFormatText(type="text")
        )

        client = self.create_openai_client()

        state = ChatCompletionStreamState(
            input_tools=openai.NOT_GIVEN,
            response_format=openai.NOT_GIVEN,
        )

        stream = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools_param,
            tool_choice=tool_choice,
            response_format=response_format_param,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            state.handle_chunk(chunk)

            # delta = chunk.choices[0].delta
            # if delta.content:
            #     print(delta.content, end="", flush=True)
            # if delta.tool_calls:
            #     print(f"\nTool Call Detected: {delta.tool_calls[0].function.name}")

        final = state.get_final_completion()
        msg = final.choices[0].message

        return AssistantMessage(
            role="assistant",
            name=assistant_name,
            content=msg.content,
            tool_calls=[t for t in msg.tool_calls] if msg.tool_calls else [],
        )
