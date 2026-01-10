#!/usr/bin/env python

import pytest
from test_core.support.common_base_models import WeatherReport
from test_core.support.helper_fake_tools import get_user_info_from_str_deps_tool, user_info_dep_type, weather_tool

from easy_agents.core.agent import Agent, SimpleContextFactory
from easy_agents.core.tool import ToolDependency, ToolDepsRegistry


@pytest.mark.asyncio
async def test_tool_less_agent(simple_router) -> None:
    agent = Agent(
        context_factory=SimpleContextFactory("You are a helpful assistant. Give short and concise answers."),
    )
    result = await agent.run("What is the official capital of Israel?", simple_router)
    assert "Jerusalem" in result


@pytest.mark.asyncio
async def test_weather_agent(simple_router) -> None:
    agent = Agent(
        context_factory=SimpleContextFactory(
            "You are a helpful weather assistant. "
            "Think step by step before making any tool calls. "
            "Always show your thinking process before a tool call."
        ),
        output_type=WeatherReport,
        tools=[weather_tool, get_user_info_from_str_deps_tool],
    )
    result = await agent.run(
        "What is the weather in the user's country's capital?",
        simple_router,
        deps=ToolDepsRegistry.from_map({user_info_dep_type: "35F living in Tel Aviv"}),
    )
    assert result.temperature == 4
    assert result.country.lower() == "israel"
    assert result.city.lower() == "jerusalem"


@pytest.mark.asyncio
async def test_agent_with_missing_deps_for_tool(simple_router) -> None:
    agent = Agent(
        context_factory=SimpleContextFactory(
            "You are a helpful weather assistant. "
            "Think step by step before making any tool calls. "
            "Always show your thinking process before a tool call."
        ),
        output_type=WeatherReport,
        tools=[weather_tool, get_user_info_from_str_deps_tool],
    )

    with pytest.raises(KeyError):
        await agent.run(
            "What is the weather in the user's country's capital?",
            simple_router,
            deps=ToolDepsRegistry.from_map(
                {ToolDependency(key="bad_key", value_type=user_info_dep_type.value_type): "35F living in Tel Aviv"}
            ),
        )


@pytest.mark.asyncio
async def test_agent_with_wrong_deps_for_tool(simple_router) -> None:
    agent = Agent(
        context_factory=SimpleContextFactory(
            "You are a helpful weather assistant. "
            "Think step by step before making any tool calls. "
            "Always show your thinking process before a tool call."
        ),
        output_type=WeatherReport,
        tools=[weather_tool, get_user_info_from_str_deps_tool],
    )

    with pytest.raises(TypeError):
        await agent.run(
            "What is the weather in the user's country's capital?",
            simple_router,
            deps=ToolDepsRegistry.from_map({ToolDependency(key=user_info_dep_type.key, value_type=int): 0}),
        )
