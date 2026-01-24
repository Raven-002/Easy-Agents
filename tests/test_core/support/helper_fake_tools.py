from easy_agents.core import RunContext, Tool, ToolDependency

from .common_base_models import WeatherQuery, WeatherResult


async def weather_tool_fn(_ctx: RunContext, _parameters: WeatherQuery) -> WeatherResult:
    return WeatherResult(temperature_c=4.0)


weather_tool = Tool[WeatherQuery, WeatherResult, None](
    name="weather_tool",
    description="get weather in a city",
    run=weather_tool_fn,
    parameters_type=WeatherQuery,
    results_type=WeatherResult,
)


async def user_info_from_deps_tool_fn(_ctx: RunContext, deps: str, _parameters: None) -> str:
    return deps


user_info_dep_type = ToolDependency("user_info", str)

get_user_info_from_str_deps_tool = Tool(
    name="user_info",
    description="get user info",
    run=user_info_from_deps_tool_fn,
    results_type=str,
    deps_type=user_info_dep_type,
)
