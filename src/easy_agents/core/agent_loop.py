import asyncio
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any

from pydantic import BaseModel

from .context import AnyChatCompletionMessage, Context, ToolCall, ToolMessage, UserMessage
from .context_refiner import ContextRefiner
from .model import AssistantResponse, Model
from .router import Router
from .run_context import RunContext, ToolDepsRegistry
from .tool import ToolAny

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


async def handle_tool_call(
    tool_call: ToolCall,
    tool: ToolAny,
    run_ctx: RunContext,
) -> ToolMessage:
    try:
        tool_result = await tool.run(run_ctx, tool_call.function.arguments)
    except Exception as e:
        tool_result = str(f"Tool failed with error: {e}.")
    print(f"Tool {tool.name}({tool_call.function.arguments}) returned: {tool_result}")
    return ToolMessage(content=str(tool_result), tool_call_id=tool_call.id, name=tool.name)


async def handle_tools(
    tool_calls: Iterable[ToolCall] | None,
    tools: list[ToolAny] | None,
    run_ctx: RunContext,
) -> Sequence[AnyChatCompletionMessage]:
    if not tool_calls:
        return []

    if not tools:
        return [UserMessage(content="There are no tools available. Do not use any tool call.")]

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
            continue
        tool_call_tasks.append(handle_tool_call(tool_call, tool, run_ctx))

    for task_future in asyncio.as_completed(tool_call_tasks):
        tool_message = await task_future
        tool_messages.append(tool_message)

    return tool_messages


async def refine_messages(ctx: Context, refiners: list[ContextRefiner] | None) -> None:
    if not refiners:
        return

    refined_messages = ctx.messages
    for refiner in refiners:
        refined_messages = await refiner.refine_new_messages(ctx.raw_messages, refined_messages)
    ctx.override_with_refined_messages(refined_messages)


async def run_agent_tools_loop(
    model: Model,
    run_context: RunContext,
    refiners: list[ContextRefiner] | None = None,
    tools: list[ToolAny] | None = None,
) -> None:
    while True:
        reply: AssistantResponse[str] = await model.chat_completion(run_context.ctx.messages, tools)
        run_context.ctx.append_message(reply.message)
        run_context.ctx.extend_messages(await handle_tools(reply.message.tool_calls, tools, run_context))
        await refine_messages(run_context.ctx, refiners)

        # When there are no tool calls left, the tools loop is finished.
        if not reply.message.tool_calls:
            break


async def run_agent_loop[OutputT: AgentLoopOutputType](
    router: Router,
    ctx: Context,
    output_type: type[OutputT] = str,  # type: ignore
    refiners: list[ContextRefiner] | None = None,
    tools: list[ToolAny] | None = None,
    deps: ToolDepsRegistry | None = None,
) -> OutputT:
    task_descriptions = context_to_task_description(ctx)
    main_model = await router.route_task(task_descriptions)
    run_context = RunContext(deps=deps or ToolDepsRegistry.empty(), ctx=ctx)
    await run_agent_tools_loop(main_model, run_context, refiners, tools)
    final_output_model = await router.route_task(context_to_final_output_task_description(task_descriptions))
    ctx.append_message(UserMessage(content="Provide the final output based on the conversation history."))
    output_message = await final_output_model.chat_completion(ctx.messages, response_format=output_type)
    return output_message.message.content
