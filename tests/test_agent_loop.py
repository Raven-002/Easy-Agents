#!/usr/bin/env python

import pytest
from pydantic import BaseModel

from easy_agents.agent.agent_loop import run_agent_loop
from easy_agents.agent.context import Context
from easy_agents.agent.model import Model
from easy_agents.agent.router import ModelId, Router
from easy_agents.agent.tool import BaseTool, RunContext

# For speed reason, use a single model
simple_models_pool: dict[ModelId, Model] = {
    "qwen3-instruct": Model(
        model_name="qwen3:4b-instruct",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        description="A very fast model for very simple tasks only.",
    ),
}

complex_models_pool: dict[ModelId, Model] = {
    "qwen3-coder": Model(
        model_name="hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_K_XL",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        description="A coding model, run at moderate speeds, good at writing code and general tasks.",
    ),
    "qwen3-instruct": Model(
        model_name="qwen3:4b-instruct",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        description="A very fast model for very simple tasks only.",
    ),
    "glm-z1-9b": Model(
        model_name="glm-z1-9b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        description="A fast model good for deep thinking and analysis",
        thinking=True,
    ),
}


@pytest.fixture(scope="module")
def simple_router(request) -> Router:
    return Router(
        models_pool=simple_models_pool,
        router_pool=["qwen3-instruct"],
    )


@pytest.fixture(scope="module")
def complex_router(request) -> Router:
    return Router(
        models_pool=complex_models_pool,
        router_pool=["qwen3-instruct"],
    )


class WeatherReport(BaseModel):
    country: str
    city: str
    temperature: float


class WeatherQuery(BaseModel):
    country: str
    city: str


class WeatherResult(BaseModel):
    temperature_c: float


async def weather_tool_fn(_ctx: RunContext[WeatherReport], _parameters: WeatherQuery) -> WeatherResult:
    return WeatherResult(temperature_c=4.0)


weather_tool = BaseTool(
    name="weather_tool",
    description="get weather in a city",
    run=weather_tool_fn,
    parameters_type=WeatherQuery,
    results_type=WeatherResult,
)


@pytest.mark.asyncio
async def test_tool_less_agent(simple_router) -> None:
    result = await run_agent_loop(
        simple_router, Context.simple("What is the capital of israel (as viewed by israeli gov)?")
    )
    assert "Jerusalem" in result


@pytest.mark.asyncio
async def test_weather_tool_sanity() -> None:
    assert 4.0 == (await weather_tool_fn(RunContext(None), WeatherQuery(country="", city=""))).temperature_c


@pytest.mark.asyncio
async def test_weather_agent(simple_router) -> None:
    result = await run_agent_loop(
        simple_router,
        Context.simple(
            "What is the weather in the capital of israel (as viewed by israeli gov)?",
            system_prompt="Plan your moves before making any tool calls. ALWAYS show your thinking process before a "
            "tool call.",
        ),
        tools=[weather_tool],
        output_type=WeatherReport,
    )
    assert result.temperature == 4
    assert result.country.lower() == "israel"
    assert result.city.lower() == "jerusalem"
