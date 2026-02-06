#!/usr/bin/env python
import pytest
from pydantic import BaseModel, Field

from easy_agents.core import (
    AssistantMessage,
    AssistantResponse,
    Context,
    Model,
    ModelTokenLimitExceededError,
    RunContext,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolMessage,
    UserMessage,
)


@pytest.fixture(autouse=True)
def skip_by_model(request: pytest.FixtureRequest, model: Model) -> None:
    skip_thinking_marker: pytest.Mark | None = request.node.get_closest_marker("skip_thinking")  # pyright: ignore
    if skip_thinking_marker and model.thinking:
        pytest.skip("Skipping thinking model")


@pytest.mark.asyncio
async def test_model_available(model: Model) -> None:
    assert await model.is_available()


@pytest.mark.asyncio
async def test_model(model: Model) -> None:
    context = Context.simple("hi", system_prompt="you need to reply with hello")
    result: AssistantResponse[str] = await model.chat_completion(context.messages)
    print(result)
    assert result.message.content.lower().find("{") == -1
    assert result.message.content.lower().find("hello") >= 0
    if model.thinking:
        assert result.message.reasoning


@pytest.mark.asyncio
async def test_response_format(model: Model) -> None:
    # The thinking part is supposed to be ignored when using response_format.
    context = Context.simple(
        "what programming languages are used in the linux kernel?", "Show a step by step thinking process"
    )

    class ResponseFormat(BaseModel):
        """A list of languages"""

        model_config = {"title": "languages_list"}
        languages: list[str]

    result = await model.chat_completion(
        context.messages,
        response_format=ResponseFormat,
    )
    print(result)
    isinstance(result.message.content, ResponseFormat)  # pyright: ignore [reportUnnecessaryIsInstance]
    assert len(result.message.content.languages) > 0


@pytest.mark.skip_thinking()
@pytest.mark.asyncio
async def test_response_format_irrelevant(model: Model) -> None:
    # The thinking part is supposed to be ignored when using response_format.
    context = Context.simple("what is the weather in TLV?", "Show a step by step thinking process")

    class ResponseFormat(BaseModel):
        languages: list[str]

    try:
        result = await model.chat_completion(context.messages, response_format=ResponseFormat, token_limit=100)
        print(result)
    except ModelTokenLimitExceededError as e:
        partial_result = e.partial_response
        print(partial_result)


@pytest.mark.skip(reason="Not supported yet")
@pytest.mark.asyncio
async def test_required_tools_irrelevant_not_needed(model: Model) -> None:
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

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[weather_tool], tool_choice="required"
    )
    print(result)
    assert result.message.content == ""
    assert result.message.tool_calls


@pytest.mark.skip(reason="Not supported yet")
@pytest.mark.asyncio
async def test_required_tools_relevant_not_needed(model: Model) -> None:
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

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[final_output_tool], tool_choice="required"
    )
    print(result)
    assert result.message.content == ""
    assert result.message.tool_calls


@pytest.mark.asyncio
async def test_required_tools_needed(model: Model) -> None:
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

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[weather_tool], tool_choice="required"
    )
    print(result)
    # Some models will give output/reasoning, while some don't. The only important part is that we get a tool call.
    assert result.message.tool_calls
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].function.name == "weather_tool"
    response = WeatherToolParams.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response in [WeatherToolParams(city="TelAviv"), WeatherToolParams(city="Tel Aviv")]


@pytest.mark.asyncio
async def test_auto_tools_needed(model: Model) -> None:
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

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[weather_tool], tool_choice="auto"
    )
    print(result)
    # Some models will give output/reasoning, while some don't. The only important part is that we get a tool call.
    assert result.message.tool_calls
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].function.name == "weather_tool"
    response = WeatherToolParams.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response in [WeatherToolParams(city="TelAviv"), WeatherToolParams(city="Tel Aviv")]


@pytest.mark.asyncio
async def test_auto_tools_relevant_not_needed(model: Model) -> None:
    context = Context.simple(
        "hi, who are you?",
        system_prompt="Think before answering. Use a tool only when there is no way to continue without it. Prefer "
        "regular output over tools. Always answer with a final output.",
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

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[final_output_tool], tool_choice="auto"
    )
    print(result)
    assert len(result.message.content) > 0
    # The model is not deterministic about using a tool call.
    # assert len(list(result["tool_calls"])) == 0


@pytest.mark.asyncio
async def test_auto_tools_irrelevant_not_needed(model: Model) -> None:
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

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[weather_tool], tool_choice="auto"
    )
    print(result)
    assert len(result.message.content) > 0
    assert result.message.tool_calls in [None, []]


@pytest.mark.asyncio
async def test_tools_in_context(model: Model) -> None:
    context = Context.simple(
        "What is the weather in the capital of israel like? what do you recommend me to wear?",
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

    context.append_message(
        AssistantMessage(
            content="I need to get the weather in Jerusalem",
            tool_calls=[
                ToolCall(
                    id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ",
                    function=ToolCallFunction(name="weather_tool", arguments='{"city": "Jerusalem"}'),
                )
            ],
        )
    )
    context.append_message(ToolMessage(content="4 Degrees Celsius", tool_call_id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ"))
    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[weather_tool], tool_choice="auto"
    )
    print(result)
    if model.thinking:
        assert result.message.reasoning
    assert result.message.content
    assert not result.message.tool_calls


@pytest.mark.asyncio
async def test_response_format_with_assistant_in_context(model: Model) -> None:
    context = Context.simple(
        "What should i wear in 4 degrees C?",
    )

    context.append_message(
        AssistantMessage(
            content="You need to wear a sweater and a scarf.",
        )
    )
    context.append_message(UserMessage(content="Answer in the format based on the messages history."))

    class WeatherReport(BaseModel):
        temperature: int
        recommended_clothes: list[str] = Field(
            default_factory=list, description="Recommended clothes, as a list of single words, without punctuation."
        )

    result: AssistantResponse[WeatherReport] = await model.chat_completion(
        context.messages, response_format=WeatherReport
    )
    print(result)
    assert result.message.content.temperature == 4
    assert "sweater" in result.message.content.recommended_clothes
    assert "scarf" in result.message.content.recommended_clothes


@pytest.mark.asyncio
async def test_tools_in_context_and_no_tools(model: Model) -> None:
    context = Context.simple(
        "What is the weather in the capital of israel like? what do you recommend me to wear?",
    )

    context.append_message(
        AssistantMessage(
            content="I need to get the weather in Jerusalem",
            tool_calls=[
                ToolCall(
                    id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ",
                    function=ToolCallFunction(name="weather_tool", arguments='{"city": "Jerusalem"}'),
                )
            ],
        )
    )
    context.append_message(ToolMessage(content="4 Degrees Celsius", tool_call_id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ"))
    context.append_message(
        AssistantMessage(content="There are 4 degrees celsius in Jerusalem. It is best to wear a sweater and a scarf.")
    )
    context.append_message(UserMessage(content="Is it cold in Jerusalem?. Answer yes or no only."))

    result: AssistantResponse[str] = await model.chat_completion(context.messages)
    print(result)
    assert result.message.content.lower().find("yes") >= 0


@pytest.mark.asyncio
async def test_response_format_with_tools_in_context(model: Model) -> None:
    context = Context.simple(
        "What is the weather in the capital of israel like? what do you recommend me to wear?",
    )

    context.append_message(
        AssistantMessage(
            content="I need to get the weather in Jerusalem",
            tool_calls=[
                ToolCall(
                    id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ",
                    function=ToolCallFunction(name="weather_tool", arguments='{"city": "Jerusalem"}'),
                )
            ],
        )
    )
    context.append_message(ToolMessage(content="4 Degrees Celsius", tool_call_id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ"))
    context.append_message(
        AssistantMessage(content="There are 4 degrees celsius in Jerusalem. It is best to wear a sweater and a scarf.")
    )
    context.append_message(UserMessage(content="Answer in the format based on the messages history."))

    class WeatherReport(BaseModel):
        city: str
        temperature: int
        reasoning: str
        recommended_clothes: list[str]

    result: AssistantResponse[WeatherReport] = await model.chat_completion(
        context.messages, response_format=WeatherReport
    )
    print(result)
    assert result.message.content.city == "Jerusalem"
    assert result.message.content.temperature == 4
    assert result.message.content.reasoning
    assert "sweater" in result.message.content.recommended_clothes
    assert "scarf" in result.message.content.recommended_clothes


@pytest.mark.asyncio
async def test_tools_with_tools_in_context_and_no_tools(model: Model) -> None:
    context = Context.simple(
        "What is the weather in the capital of israel like? what do you recommend me to wear?",
    )

    context.append_message(
        AssistantMessage(
            content="I need to get the weather in Jerusalem",
            tool_calls=[
                ToolCall(
                    id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ",
                    function=ToolCallFunction(name="weather_tool", arguments='{"city": "Jerusalem"}'),
                )
            ],
        )
    )
    context.append_message(ToolMessage(content="4 Degrees Celsius", tool_call_id="aGDfqm6prQ0ve0A1Mz6oLIlMOQPjWuzZ"))
    context.append_message(
        AssistantMessage(content="There are 4 degrees celsius in Jerusalem. It is best to wear a sweater and a scarf.")
    )
    context.append_message(UserMessage(content="Answer in the format based on the messages history."))

    class WeatherReport(BaseModel):
        city: str
        temperature: int
        reasoning: str
        recommended_clothes: list[str]

    async def weather_tool_fn(_ctx: RunContext, params: WeatherReport) -> WeatherReport:
        print(params)
        return params

    weather_report_tool = Tool[WeatherReport, WeatherReport, None](
        "weather_report_tool", "Report Weather", weather_tool_fn, WeatherReport, WeatherReport
    )

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[weather_report_tool], tool_choice="required"
    )
    print(result)
    assert result.message.tool_calls
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].function.name == "weather_report_tool"
    response = WeatherReport.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response.city == "Jerusalem"
    assert response.temperature == 4
    assert response.reasoning
    assert "sweater" in response.recommended_clothes
    assert "scarf" in response.recommended_clothes


@pytest.mark.asyncio
async def test_tools_with_content(model: Model) -> None:
    context = Context.simple(
        "What is the weather in the capital of israel like? what do you recommend me to wear?",
        system_prompt="Show a step by step thinking process. Think before making any tool call. Do not provide tool "
        "calls without planning. You MUST provide a summerized plan before making any tool calls. Your internal "
        "thought process will be erased after each tool call. Show your thinking process as final output before a tool "
        "call. Always answer with a final output.",
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

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[weather_tool], tool_choice="auto"
    )
    print(result)
    if model.thinking:
        assert result.message.reasoning
    assert result.message.content
    assert result.message.tool_calls
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].function.name == "weather_tool"
    response = WeatherToolParams.model_validate_json(result.message.tool_calls[0].function.arguments, strict=True)
    assert response in [WeatherToolParams(city="Jerusalem"), WeatherToolParams(city="Tel Aviv")]


@pytest.mark.asyncio
async def test_thinking(model: Model) -> None:
    context = Context.simple(
        "what is the largest country in size?", system_prompt="Show a step by step thinking process"
    )
    result: AssistantResponse[str] = await model.chat_completion(context.messages)
    print(result)
    if model.thinking:
        assert result.message.reasoning
    else:
        assert not result.message.reasoning
        assert result.message.content.lower().find("step") >= 0


@pytest.mark.asyncio
async def test_tools_description(model: Model) -> None:
    context = Context.simple(
        "Hello!", system_prompt="Reply the user with a greeting with their name. Use the correct tool."
    )

    class ToolParamsA(BaseModel):
        """Get the user's info."""

        p1: str = Field(..., description="What to get, 'name' or 'age'")
        p2: str = Field(..., description="Reasoning for the request.")

    class ToolParamsB(BaseModel):
        """crash the program. Should never be used."""

        # Not all tools can see the params docstring, so it is also added in the field to make sure it is seen.
        p1: str = Field(..., description="Reasoning for the crash request. This tool should never be used.")

    async def tool_a_fn(_ctx: RunContext, params: ToolParamsA) -> str:
        print(params)
        return "hi"

    async def tool_b_fn(_ctx: RunContext, params: ToolParamsB) -> str:
        print(params)
        return "hi"

    tool_a = Tool[ToolParamsA, str, None]("tool_a", "get info", tool_a_fn, ToolParamsA, str)

    tool_b = Tool[ToolParamsB, str, None]("tool_b", "get user name", tool_b_fn, ToolParamsB, str)

    result: AssistantResponse[str] = await model.chat_completion(
        context.messages, tools=[tool_a, tool_b], tool_choice="auto"
    )
    print(result)
    assert result.message.tool_calls
    assert len(result.message.tool_calls) == 1
    assert result.message.tool_calls[0].function.name == "tool_a"
