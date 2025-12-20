from dataclasses import dataclass
from typing import Any

import litellm

from ai_cr.utils.ai_utils.tools import AbstractTool
from ai_cr.utils.ai_utils.types.context import Context
from ai_cr.utils.common import JsonType
from ai_cr.utils.logging_utils import dlog


@dataclass
class AiResponse:
    role: str
    content: str
    tool_calls: list[Any]  # Contains litellm.utils.ChatCompletionMessageToolCall objects


class AiRunner:
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

    @classmethod
    def create_runner(cls, runner_params) -> "AiRunner":
        return cls(**runner_params)

    def generate_response(
        self,
        prompt: str,
        system_prompt: str,
        context: Context,
        tools: list[type[AbstractTool]],
        response_format: JsonType,
        thinking: bool = False,
    ) -> AiResponse:
        # Prepare the message history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context.messages)
        messages.append({"role": "user", "content": prompt})

        # Convert Python functions to JSON schemas for LiteLLM
        tools_dicts = [t.schema() for t in tools] if tools else None

        dlog(
            f"Running AI with:\n - Messages: {'\n'.join([str(m) for m in messages])}\n - Tools: {tools_dicts}",
            markup=False,
        )

        # Start streaming completion
        response_stream = litellm.completion(
            model=self.model_name,
            messages=messages,
            tools=tools_dicts,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
            response_format=response_format,
            extra_body={"enable_thinking": thinking},
        )

        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
        #     # Live stream content to the terminal
        #     content = chunk.choices[0].delta.content
        #     if content:
        #         print(content, end="", flush=True)
        # print()

        # Reconstruct the full message from chunks
        full_response = litellm.stream_chunk_builder(chunks, messages=messages)
        message_obj = full_response.choices[0].message

        dlog(f"AI response:\n - Content: {message_obj.content}\n\n - ToolCalls: {message_obj.tool_calls}", markup=False)

        return AiResponse(
            role="assistant", content=message_obj.content or "", tool_calls=getattr(message_obj, "tool_calls", []) or []
        )
