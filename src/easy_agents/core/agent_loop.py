import asyncio
from collections.abc import Coroutine, Iterable
from typing import Any

from pydantic import BaseModel

from .context import Context, ToolCall, ToolMessage, UserMessage
from .model import Model
from .router import Router
from .tool import Tool, RunContext

type AgentLoopOutputType = BaseModel | str


def _context_to_messages_string(ctx: Context) -> str:
    return f"[{', '.join([str(m.model_dump()) for m in ctx.messages])}]"


def context_to_task_description(ctx: Context) -> str:
    return f"This agent is tasked with {_context_to_messages_string(ctx)}."


def context_to_final_output_task_description(task_description: str) -> str:
    return (
        f"The task is to provide the final output for the following agent: {task_description}\n\n"
        "The model will need to extract the final output from the conversation history, and provide it as a structured "
        "output."
    )


async def handle_tool_call[T](
    tool_call: ToolCall,
    tool: Tool,
    run_ctx: RunContext[T] = None,
) -> ToolMessage:
    tool_result = await tool.run(run_ctx, tool_call.function.arguments)
    return ToolMessage(content=str(tool_result), tool_call_id=tool_call.id, name=tool.name)


async def handle_tools[T](
    tool_calls: Iterable[ToolCall],
    tools: Iterable[Tool],
    run_ctx: RunContext[T] = None,
) -> list[ToolMessage]:
    tools_map = {tool.name: tool for tool in tools}
    tool_messages: list[ToolMessage] = []
    tool_call_tasks: list[Coroutine[Any, Any, ToolMessage]] = []

    for tool_call in tool_calls:
        tool = tools_map.get(tool_call.function.name)
        if tool is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Tool {tool_call.function.name} not found.",
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                )
            )
        tool_call_tasks.append(handle_tool_call(tool_call, tool, run_ctx))

    for task_future in asyncio.as_completed(tool_call_tasks):
        tool_message = await task_future
        tool_messages.append(tool_message)

    return tool_messages


async def run_agent_tools_loop[DepsT: Any](
    model: Model,
    ctx: Context,
    tools: Iterable[Tool] | None = None,
    deps: DepsT = None,
) -> None:
    while True:
        reply = model.chat_completion(ctx.messages, tools)
        ctx.messages.append(reply.message)

        # When there are no tool calls left, the tools loop is finished.
        if not reply.message.tool_calls:
            return

        if not tools:
            ctx.messages.append(UserMessage(content="There are no tools available. Do not use any tool call."))

        ctx.messages.extend(await handle_tools(reply.message.tool_calls, tools, deps))


async def run_agent_loop[OutputT: AgentLoopOutputType, DepsT: Any](
    router: Router,
    ctx: Context,
    output_type: type[OutputT] = str,
    tools: Iterable[Tool] | None = None,
    deps: DepsT = None,
) -> OutputT:
    task_descriptions = context_to_task_description(ctx)
    main_model = router.route_task(task_descriptions)
    await run_agent_tools_loop(main_model, ctx, tools, deps)
    final_output_model = router.route_task(context_to_final_output_task_description(task_descriptions))
    ctx.messages.append(UserMessage(content="Provide the final output based on the conversation history."))
    output_message = final_output_model.chat_completion(ctx.messages, response_format=output_type)
    return output_message.message.content
