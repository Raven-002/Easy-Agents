from typing import Any

from easy_agents.core.tool import RunContext, Tool

from .common_base_models import WeatherQuery, WeatherResult


async def weather_tool_fn(_ctx: RunContext[Any], _parameters: WeatherQuery) -> WeatherResult:
    return WeatherResult(temperature_c=4.0)


weather_tool = Tool(
    name="weather_tool",
    description="get weather in a city",
    run=weather_tool_fn,
    parameters_type=WeatherQuery,
    results_type=WeatherResult,
)


async def user_info_from_deps_tool_fn(_ctx: RunContext[Any], deps: str, _parameters: None) -> str:
    return deps


get_user_info_from_str_deps_tool = Tool(
    name="user_info",
    description="get user info",
    run=user_info_from_deps_tool_fn,
    results_type=str,
    deps_type=str,
    app_deps_type=str,
    deps_extractor=lambda deps: deps,
)
