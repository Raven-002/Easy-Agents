#!/usr/bin/env python

import pytest
from test_core.support.common_base_models import WeatherReport
from test_core.support.helper_fake_tools import get_user_info_from_str_deps_tool, weather_tool

from easy_agents.core.agent import Agent, SimpleContextFactory


@pytest.mark.asyncio
async def test_tool_less_agent(simple_router) -> None:
    agent = Agent(
        router=simple_router,
        context_factory=SimpleContextFactory("You are a helpful assistant. Give short and concise answers."),
    )
    result = await agent.run("What is the official capital of Israel?", None)
    assert "Jerusalem" in result


@pytest.mark.asyncio
async def test_weather_agent(simple_router) -> None:
    agent = Agent(
        simple_router,
        context_factory=SimpleContextFactory(
            "You are a helpful weather assistant. "
            "Think step by step before making any tool calls. "
            "Always show your thinking process before a tool call."
        ),
        output_type=WeatherReport,
        tools=[weather_tool, get_user_info_from_str_deps_tool],
        deps_type=str,
    )
    result = await agent.run("What is the weather in the user's country's capital?", deps="35F living in Tel Aviv")
    assert result.temperature == 4
    assert result.country.lower() == "israel"
    assert result.city.lower() == "jerusalem"
