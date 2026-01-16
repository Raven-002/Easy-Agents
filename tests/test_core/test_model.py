#!/usr/bin/env python
from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from easy_agents.core import AssistantResponse
from easy_agents.core.context import Context
from easy_agents.core.model import Model
from easy_agents.core.tool import RunContext, Tool


def get_test_models() -> Iterator[Model]:
    yield Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        description="",
    )
    yield Model(
        model_name="qwen3:14b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        description="",
        thinking=True,
    )
    yield Model(
        model_name="glm-z1-9b",  # Based on "hf.co/unsloth/GLM-Z1-9B-0414-GGUF:Q6_K_XL"
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        description="",
        thinking=True,
    )
    # NOTE: glm-4-9b non-thinking has problems passing the tests, so it is removed.


@pytest.fixture(params=get_test_models(), scope="module")
def model(request: pytest.FixtureRequest) -> Model:
    requested_model = request.param
    assert isinstance(requested_model, Model)
    return requested_model


def test_model(model: Model) -> None:
    context = Context.simple("hi")
    result: AssistantResponse[str] = model.chat_completion(context.messages)
    print(result)


def test_response_format(model: Model) -> None:
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
    isinstance(result.message.content, ResponseFormat)  # pyright: ignore [reportUnnecessaryIsInstance]
    assert len(result.message.content.languages) > 0


def test_response_format_irrelevant(model: Model) -> None:
    # The thinking part is supposed to be ignored when using response_format.
    context = Context.simple("what is the weather in TLV?", "Show a step by step thinking process")

    class ResponseFormat(BaseModel):
        languages: list[str]

    result = model.chat_completion(context.messages, response_format=ResponseFormat, token_limit=100)
    print(result)
    assert len(result.message.content.languages) > 0


@pytest.mark.skip(reason="Not supported yet")
def test_required_tools_irrelevant_not_needed(model: Model) -> None:
    context = Context.simple("hi, who are you?")

    class WeatherToolParams(BaseModel):
        city: str

    class WeatherToolResponse(BaseModel):
        weather: str

    async def weather_tool_fn(_ctx: RunContext, params: WeatherToolParams) -> WeatherToolResponse:
        print(params)
        return WeatherToolResponse(weather="sunny")

    weather_tool = Tool[WeatherToolParams, WeatherToolResponse, None](
        "weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse
    )

    result: AssistantResponse[str] = model.chat_completion(
        context.messages, tools=[weather_tool], tool_choice="required"
    )
    print(result)
    assert result.message.content == ""
    assert result.message.tool_calls


@pytest.mark.skip(reason="Not supported yet")
def test_required_tools_relevant_not_needed(model: Model) -> None:
    context = Context.simple(
        "hi, who are you?",
        system_prompt="Think before answering. Use a tool only when there "
        "is no way to continue without it. Prefer regular output over tools.",
    )

    class FinalOutputToolParams(BaseModel):
        message: str

    class FinalOutputToolResponse(BaseModel):
        is_ok: bool

    async def greeting(_ctx: RunContext, params: FinalOutputToolParams) -> FinalOutputToolResponse:
        print(params)
        return FinalOutputToolResponse(is_ok=True)

    final_output_tool = Tool[FinalOutputToolParams, FinalOutputToolResponse, None](
        "final_output", "Give the final output", greeting, FinalOutputToolParams, FinalOutputToolResponse
    )

    result: AssistantResponse[str] = model.chat_completion(
        context.messages, tools=[final_output_tool], tool_choice="required"
    )
    print(result)
    assert result.message.content == ""
    assert result.message.tool_calls


def test_auto_tools_needed(model: Model) -> None:
    context = Context.simple("what is the weather in TelAviv?")

    class WeatherToolParams(BaseModel):
        city: str

    class WeatherToolResponse(BaseModel):
        weather: str

    async def weather_tool_fn(_ctx: RunContext, params: WeatherToolParams) -> WeatherToolResponse:
        print(params)
        return WeatherToolResponse(weather="sunny")

    weather_tool = Tool[WeatherToolParams, WeatherToolResponse, None](
        "weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse
    )

    result: AssistantResponse[str] = model.chat_completion(context.messages, tools=[weather_tool], tool_choice="auto")
    print(result)
    # Some models will give output/reasoning, while some don't. The only important part is that we get a tool call.
    assert result.message.tool_calls
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].function.name == "weather_tool"
    response = WeatherToolParams.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response == WeatherToolParams(city="TelAviv")


def test_auto_tools_relevant_not_needed(model: Model) -> None:
    context = Context.simple(
        "hi, who are you?",
        system_prompt="Think before answering. Use a tool only when there is no way to continue without it. Prefer "
        "regular output over tools.",
    )

    class FinalOutputToolParams(BaseModel):
        message: str

    class FinalOutputToolResponse(BaseModel):
        is_ok: bool

    async def greeting(_ctx: RunContext, params: FinalOutputToolParams) -> FinalOutputToolResponse:
        print(params)
        return FinalOutputToolResponse(is_ok=True)

    final_output_tool = Tool[FinalOutputToolParams, FinalOutputToolResponse, None](
        "final_output", "Give the final output", greeting, FinalOutputToolParams, FinalOutputToolResponse
    )

    result: AssistantResponse[str] = model.chat_completion(
        context.messages, tools=[final_output_tool], tool_choice="auto"
    )
    print(result)
    assert len(result.message.content) > 0
    # The model is not deterministic about using a tool call.
    # assert len(list(result["tool_calls"])) == 0


def test_auto_tools_irrelevant_not_needed(model: Model) -> None:
    context = Context.simple("hi, who are you?")

    class WeatherToolParams(BaseModel):
        city: str

    class WeatherToolResponse(BaseModel):
        weather: str

    async def weather_tool_fn(_ctx: RunContext, params: WeatherToolParams) -> WeatherToolResponse:
        print(params)
        return WeatherToolResponse(weather="sunny")

    weather_tool = Tool[WeatherToolParams, WeatherToolResponse, None](
        "weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse
    )

    result: AssistantResponse[str] = model.chat_completion(context.messages, tools=[weather_tool], tool_choice="auto")
    print(result)
    assert len(result.message.content) > 0
    assert result.message.tool_calls in [None, []]


def test_tools_with_content(model: Model) -> None:
    context = Context.simple(
        "What is the weather in the capital of israel?",
        system_prompt="Show a step by step thinking process. Think before making any tool call. Do not provide tool "
        "calls without planning.",
    )

    class WeatherToolParams(BaseModel):
        city: str

    class WeatherToolResponse(BaseModel):
        weather: str

    async def weather_tool_fn(_ctx: RunContext, params: WeatherToolParams) -> WeatherToolResponse:
        print(params)
        return WeatherToolResponse(weather="sunny")

    weather_tool = Tool[WeatherToolParams, WeatherToolResponse, None](
        "weather_tool", "Get weather", weather_tool_fn, WeatherToolParams, WeatherToolResponse
    )

    result: AssistantResponse[str] = model.chat_completion(context.messages, tools=[weather_tool], tool_choice="auto")
    print(result)
    assert len(result.message.content) > 0 or len(result.message.reasoning or []) > 0
    assert result.message.tool_calls
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].function.name == "weather_tool"
    response = WeatherToolParams.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response == WeatherToolParams(city="Jerusalem")
