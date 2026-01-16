#!/usr/bin/env python


import pytest

from easy_agents.core import Router
from easy_agents.core.agent_loop import run_agent_loop
from easy_agents.core.context import Context
from tests.test_core.support.common_base_models import WeatherReport
from tests.test_core.support.helper_fake_tools import weather_tool


@pytest.mark.asyncio
async def test_tool_less_agent(simple_router: Router) -> None:
    result: str = await run_agent_loop(
        simple_router, Context.simple("What is the capital of israel (as viewed by israeli gov)?")
    )
    assert "Jerusalem" in result


@pytest.mark.asyncio
async def test_weather_agent(simple_router: Router) -> None:
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
