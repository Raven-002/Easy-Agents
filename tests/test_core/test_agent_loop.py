#!/usr/bin/env python
from collections.abc import Sequence

import pytest

from easy_agents.core import Context, ContextRefiner, Router
from easy_agents.core.agent_loop import run_agent_loop
from easy_agents.core.context import AnyChatCompletionMessage, AssistantMessage
from easy_agents.core.context_refiner import MessagesEndsWithAssistantMessage
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


class FakeContextRefiner(ContextRefiner):
    def __init__(
        self,
        refined_messages_to_inject: Sequence[AnyChatCompletionMessage] | None = None,
        assistant_message_to_inject: AssistantMessage[str] | None = None,
    ) -> None:
        self.refined_messages_to_inject = refined_messages_to_inject
        self.assistant_message_to_inject = assistant_message_to_inject
        super().__init__()

    async def refine_new_messages(
        self,
        router: Router,
        raw_messages: Sequence[AnyChatCompletionMessage],
        refined_messages: Sequence[AnyChatCompletionMessage],
    ) -> Sequence[AnyChatCompletionMessage]:
        if self.refined_messages_to_inject is None:
            return refined_messages
        return self.refined_messages_to_inject

    async def refine_pre_tool_assistant_message(
        self,
        router: Router,
        raw_messages: MessagesEndsWithAssistantMessage[str],
        refined_messages: MessagesEndsWithAssistantMessage[str],
    ) -> AssistantMessage[str]:
        if self.assistant_message_to_inject is None:
            return refined_messages[-1]
        return self.assistant_message_to_inject


@pytest.mark.asyncio
async def test_pre_tool_refiners(simple_router: Router) -> None:
    result = await run_agent_loop(
        simple_router,
        Context.simple(
            "What is the weather in the capital of israel (as viewed by israeli gov)?",
            system_prompt="Plan your moves before making any tool calls. ALWAYS show your thinking process before a "
            "tool call.",
        ),
        tools=[weather_tool],
        refiners=[FakeContextRefiner(assistant_message_to_inject=AssistantMessage(content="Jerusalem, israel: 44C"))],
        output_type=WeatherReport,
    )
    assert result.temperature == 44
    assert result.country.lower() == "israel"
    assert result.city.lower() == "jerusalem"


@pytest.mark.asyncio
async def test_full_refiners_override(simple_router: Router) -> None:
    result = await run_agent_loop(
        simple_router,
        Context.simple(
            "What is the weather in the capital of israel (as viewed by israeli gov)?",
            system_prompt="Plan your moves before making any tool calls. ALWAYS show your thinking process before a "
            "tool call.",
        ),
        tools=[weather_tool],
        refiners=[
            FakeContextRefiner(
                refined_messages_to_inject=[AssistantMessage(content="America, washington, 40C")],
                assistant_message_to_inject=AssistantMessage(content="Jerusalem, israel: 14C"),
            )
        ],
        output_type=WeatherReport,
    )
    assert result.temperature == 40
    assert result.country.lower() in ["america", "us", "united states", "united states of america"]
    assert result.city.lower() == "washington"
