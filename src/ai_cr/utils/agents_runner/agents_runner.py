from typing import Any

from pydantic_ai import Agent, AgentRunResultEvent, FinalResultEvent, PartEndEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
)
from rich.panel import Panel

from ai_cr.logger.logging_utils import get_console, is_debug, is_verbose, status


def should_print_tokens() -> bool:
    return is_debug()


def should_print_event_titles() -> bool:
    return is_verbose()


def should_print_event_content() -> bool:
    return is_debug()


def print_token(t: str) -> None:
    if should_print_tokens():
        get_console().print(t, end="")


def print_event(e: Any) -> None:
    if isinstance(e, PartDeltaEvent):
        # if hasattr(e.delta, "content_delta"):
        print_token(e.delta.content_delta)

    elif isinstance(e, FunctionToolCallEvent):
        if not should_print_event_titles():
            return
        get_console().rule(f"Calling Tool: {e.part.tool_name}")
        if should_print_event_content():
            get_console().print(f"Tool Args: {e.part.args}")

    elif isinstance(e, FunctionToolResultEvent):
        if not should_print_event_titles():
            return
        if should_print_event_content():
            get_console().print(f"Tool Response: {e.result.content}")
        get_console().rule(f"Tool Response: {e.result.tool_name}")

    elif isinstance(e, PartStartEvent):
        if should_print_event_titles():
            get_console().rule(title=f"Event Started: {type(e.part).__name__}")
        if hasattr(e.part, "content") and e.part.content is not None:
            print_token(e.part.content)

    elif isinstance(e, PartEndEvent):
        if type(e.part) not in [ToolCallPart]:
            print_token("\n")
        if not should_print_event_titles():
            return
        if should_print_event_content():
            get_console().print(f"{e}")
            get_console().rule(title="Event Ended")
        else:
            get_console().rule(title="Event Ended")

    elif isinstance(e, FinalResultEvent):
        if should_print_event_titles():
            get_console().rule("Final Results")

    elif isinstance(e, AgentRunResultEvent):
        if should_print_event_titles():
            get_console().rule("Final Agent Results")
        if should_print_event_content():
            results = e.result.output
            get_console().print(Panel(str(results)))
    else:
        raise NotImplementedError(f"Missing implementation for event of type: {type(e)} - {e}")


async def run_agent[T](agent: Agent[T], prompt: str, deps: T) -> Any:
    results: Any | None = None
    if should_print_tokens():
        stream_events = agent.run_stream_events(prompt, deps=deps)
        async for event in stream_events:
            print_event(event)
            if isinstance(event, AgentRunResultEvent):
                results = event.result.output
    else:
        with status("Running LLM..."):
            stream_events = agent.run_stream_events(prompt, deps=deps)
            async for event in stream_events:
                print_event(event)
                if isinstance(event, AgentRunResultEvent):
                    results = event.result.output
    assert results is not None
    return results
