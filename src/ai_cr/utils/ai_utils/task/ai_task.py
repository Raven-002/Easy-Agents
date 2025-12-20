import json
from enum import Enum

from pydantic import TypeAdapter

from ai_cr.utils.ai_utils.types.context import Context

from ...common import JsonType
from .. import AiRunner
from ..tools import AbstractTool


class TaskState(Enum):
    INITIAL_PLANNING = "initial_planning"
    TOOL_CALL = "tool_call"
    CONTINUE_PLANNING = "continue_planning"
    FINAL_OUTPUT = "final_output"
    ERROR = "error"
    FINISHED = "finished"


class AiTask[T]:
    def __init__(
        self,
        prompt: str,
        output_schema: type[T],
        tools: list[type[AbstractTool]] | None = None,
        system_prompt: str = "You are a specialized agent. Use tools if needed, then respond with the final data.",
    ):
        self.prompt = prompt
        self.output_schema = output_schema
        self.tools = tools or []
        self.system_prompt = system_prompt
        self._adapter = TypeAdapter(output_schema)

    def _get_response_schema(self) -> JsonType:
        """Generates the OpenAI-compatible JSON schema from the dataclass."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.output_schema.__name__.lower(),
                "strict": True,
                "schema": self._adapter.json_schema(),
            },
        }

    @staticmethod
    def _get_planning_schema() -> JsonType:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "planning_stage",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "planning": {"type": "string", "description": "High-level planning and decision making"},
                        "need_tool_call": {
                            "type": "boolean",
                            "description": "Whether a tool call will be required to complete the task",
                        },
                        "suggested_tools": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "Name of the tool"},
                                    "suggested_arguments": {
                                        "type": "string",
                                        "description": "plain text of suggested arguments",
                                    },
                                },
                            },
                            "description": "Tools that should be called to continue the task",
                        },
                    },
                    "required": ["planning", "need_tool_call"],
                    "additionalProperties": False,
                },
            },
        }

    @property
    def planning_system_prompt(self) -> str:
        return self.system_prompt + "\n\nProvide high-level planning and decision making.\nDo not provide tool calls"

    @property
    def tool_call_system_prompt(self) -> str:
        return (
            "Only provide tool calls in the following format: "
            '{ "name": "tool_name", "arguments": { "arg1": "value1", ... }"}, as described by the tools.'
        )

    @property
    def final_output_system_prompt(self) -> str:
        return (
            self.system_prompt + "\n\nProvide the final structured output based on the planning and tool calls."
            "\nDo not provide tool calls."
        )

    def run(self, runner: AiRunner, context: Context) -> T:
        """The Agent Loop: Handles tool calls until the final structured output is reached."""
        tool_map = {t.__name__: t for t in self.tools}
        state = TaskState.INITIAL_PLANNING

        while True:
            match state:
                case TaskState.INITIAL_PLANNING:
                    response = runner.generate_response(
                        prompt=self.prompt,
                        system_prompt=self.planning_system_prompt,
                        context=context,
                        tools=self.tools,
                        response_format=self._get_planning_schema(),
                    )
                    plan_result = json.loads(response.content)
                    state = TaskState.TOOL_CALL if plan_result["need_tool_call"] else TaskState.FINAL_OUTPUT
                    context.add_user_message(self.prompt)
                    context.add_assistant_message(response.content)
                case TaskState.TOOL_CALL:
                    prompt = "Provide tool calls based on the planning."
                    response = runner.generate_response(
                        prompt=prompt,
                        system_prompt=self.tool_call_system_prompt,
                        context=context,
                        tools=self.tools,
                        response_format=None,
                    )
                    state = TaskState.CONTINUE_PLANNING
                    context.add_user_message(prompt)
                    context.add_assistant_message(response.content, response.tool_calls)
                    if response.tool_calls:
                        for call in response.tool_calls:
                            tool = tool_map.get(call.function.name)
                            if not tool:
                                raise ValueError(f"Tool {call.function.name} not found.")
                            args = json.loads(call.function.arguments)
                            result = tool.run(**args)
                            context.add_tool_message(content=str(result), tool_call_id=call.id)
                    else:
                        tool_call = json.loads(response.content)
                        tool = tool_map.get(tool_call["name"])
                        if not tool:
                            raise ValueError(f"Tool {tool_call['name']} not found.")
                        args = tool_call["arguments"]
                        result = tool.run(**args)
                        context.add_tool_message(content=str(result), tool_call_id="last_tool_call")
                case TaskState.CONTINUE_PLANNING:
                    prompt = "Continue planning based on the tool calls."
                    response = runner.generate_response(
                        prompt=prompt,
                        system_prompt=self.planning_system_prompt,
                        context=context,
                        tools=self.tools,
                        response_format=self._get_planning_schema(),
                    )
                    plan_result = json.loads(response.content)
                    state = TaskState.TOOL_CALL if plan_result["need_tool_call"] else TaskState.FINAL_OUTPUT
                    context.add_user_message(prompt)
                    context.add_assistant_message(response.content)
                case TaskState.FINAL_OUTPUT:
                    response = runner.generate_response(
                        prompt="Provide the final structured output based on the planning and tool calls.",
                        system_prompt=self.final_output_system_prompt,
                        context=context,
                        tools=self.tools,
                        response_format=self._get_response_schema(),
                    )
                    state = TaskState.FINISHED
                    return self._adapter.validate_python(json.loads(response.content))
                case _:  # type: ignore
                    raise ValueError(f"Invalid state: {state}")
