#!/usr/bin/env python
from typing import Any

from pydantic import BaseModel

from easy_agents.agent.context import Context
from easy_agents.agent.model import Model
from easy_agents.agent.tool import BaseTool, RunContext


def test_model() -> None:
    model = Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )
    context = Context.simple("hi")
    result = model.chat_completion(context.messages)
    print(result)


def test_response_format() -> None:
    model = Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )
    # The thinking part is supposed to be ignored when using response_format.
    context = Context.simple(
        "what programming languages are used in the linux kernel?", "Show a step by step thinking process"
    )

    class ResponseFormat(BaseModel):
        """A list of languages"""

        model_config = {"title": "languages_list"}
        languages: list[str]

    result = model.chat_completion(
        context.messages,
        response_format=ResponseFormat,
    )
    print(result)
    isinstance(result.message.content, ResponseFormat)
    assert len(result.message.content.languages) > 0


def test_response_format_irrelevant() -> None:
    model = Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )
    # The thinking part is supposed to be ignored when using response_format.
    context = Context.simple("what is the weather in TLV?", "Show a step by step thinking process")

    class ResponseFormat(BaseModel):
        languages: list[str]

    result = model.chat_completion(
        context.messages,
        response_format=ResponseFormat,
    )
    print(result)
    assert len(result.message.content.languages) > 0


# The model used does not support required tools, so this test is skipped.
# def test_required_tools_irrelevant_not_needed() -> None:
#     model = Model(
#         model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
#         api_base="http://localhost:11434/v1",
#         api_key="ollama",
#     )
#     context = Context.simple("hi, who are you?")
#
#     class WeatherToolParams(BaseModel):
#         city: str
#
#     class WeatherToolResponse(BaseModel):
#         weather: str
#
#     async def weather_tool_fn(_ctx: RunContext[Any], params: WeatherToolParams) -> WeatherToolResponse:
#         print(params)
#         return WeatherToolResponse(weather="sunny")
#
#     weather_tool = BaseTool("weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse)
#
#     result = model.chat_completion(context.messages, tools=[weather_tool], tool_choice="required")
#     print(result)
#     assert result["content"] == ""
#     assert len(list(result["tool_calls"])) > 0


# def test_required_tools_relevant_not_needed() -> None:
#     model = Model(
#         model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
#         api_base="http://localhost:11434/v1",
#         api_key="ollama",
#     )
#     context = Context.simple(
#         "hi, who are you?",
#         system_prompt="Think before answering. Use a tool only when there "
#         "is no way to continue without it. Prefer regular output over tools.",
#     )
#
#     class FinalOutputToolParams(BaseModel):
#         message: str
#
#     class FinalOutputToolResponse(BaseModel):
#         is_ok: bool
#
#     async def greeting(_ctx: RunContext[Any], params: FinalOutputToolParams) -> FinalOutputToolResponse:
#         print(params)
#         return FinalOutputToolResponse(is_ok=True)
#
#     final_output_tool = BaseTool(
#         "final_output", "Give the final output", greeting, FinalOutputToolParams, FinalOutputToolResponse
#     )
#
#     result = model.chat_completion(context.messages, tools=[final_output_tool], tool_choice="required")
#     print(result)
#     assert result["content"] == ""
#     assert len(list(result["tool_calls"])) > 0


def test_auto_tools_needed() -> None:
    model = Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )
    context = Context.simple("what is the weather in TelAviv?")

    class WeatherToolParams(BaseModel):
        city: str

    class WeatherToolResponse(BaseModel):
        weather: str

    async def weather_tool_fn(_ctx: RunContext[Any], params: WeatherToolParams) -> WeatherToolResponse:
        print(params)
        return WeatherToolResponse(weather="sunny")

    weather_tool = BaseTool("weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse)

    result = model.chat_completion(context.messages, tools=[weather_tool], tool_choice="auto")
    print(result)
    assert result.message.content == ""
    assert len(list(result.message.tool_calls)) == 1
    assert result.message.tool_calls[0].function.name == "weather_tool"
    response = WeatherToolParams.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response == WeatherToolParams(city="TelAviv")


def test_auto_tools_relevant_not_needed() -> None:
    model = Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )
    context = Context.simple(
        "hi, who are you?",
        system_prompt="Think before answering. Use a tool only when there is no way to continue without it. Prefer "
        "regular output over tools.",
    )

    class FinalOutputToolParams(BaseModel):
        message: str

    class FinalOutputToolResponse(BaseModel):
        is_ok: bool

    async def greeting(_ctx: RunContext[Any], params: FinalOutputToolParams) -> FinalOutputToolResponse:
        print(params)
        return FinalOutputToolResponse(is_ok=True)

    final_output_tool = BaseTool(
        "final_output", "Give the final output", greeting, FinalOutputToolParams, FinalOutputToolResponse
    )

    result = model.chat_completion(context.messages, tools=[final_output_tool], tool_choice="auto")
    print(result)
    assert len(result.message.content) > 0
    # The model is not deterministic about using a tool call.
    # assert len(list(result["tool_calls"])) == 0


def test_auto_tools_irrelevant_not_needed() -> None:
    model = Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )
    context = Context.simple("hi, who are you?")

    class WeatherToolParams(BaseModel):
        city: str

    class WeatherToolResponse(BaseModel):
        weather: str

    async def weather_tool_fn(_ctx: RunContext[Any], params: WeatherToolParams) -> WeatherToolResponse:
        print(params)
        return WeatherToolResponse(weather="sunny")

    weather_tool = BaseTool("weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse)

    result = model.chat_completion(context.messages, tools=[weather_tool], tool_choice="auto")
    print(result)
    assert len(result.message.content) > 0
    assert len(list(result.message.tool_calls)) == 0


def test_tools_with_content() -> None:
    model = Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
    )
    context = Context.simple(
        "What is the weather in the capital of israel?",
        system_prompt="Show a step by step thinking process. Think before making any tool call.",
    )

    class WeatherToolParams(BaseModel):
        city: str

    class WeatherToolResponse(BaseModel):
        weather: str

    async def weather_tool_fn(_ctx: RunContext[Any], params: WeatherToolParams) -> WeatherToolResponse:
        print(params)
        return WeatherToolResponse(weather="sunny")

    weather_tool = BaseTool("weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse)

    result = model.chat_completion(context.messages, tools=[weather_tool], tool_choice="auto")
    print(result)
    assert len(result.message.content) > 0
    assert len(list(result.message.tool_calls)) == 1
    assert result.message.tool_calls[0].function.name == "weather_tool"
    response = WeatherToolParams.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response == WeatherToolParams(city="Jerusalem")
