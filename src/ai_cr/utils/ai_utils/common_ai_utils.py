from ai_cr.utils.ai_utils.tools.base_tool import BaseTool

base_system_prompt_output_description = """You have a scratchpad for thinking, and a final output. Your scratchpad is
erased after the final output of this turn is generated, but preserved between tool calls.

your entire output is split by using tags:
```
<scratchpad>
</scratchpad>
<output>
</output>
```

In the scratchpad, you can use simple language, markdown, everything other than tags is considered scratchpad.

Your output is a JSON object with one of the following formats:

For tool calls:
<output>
{
    "type": "tool",
    "tool_name": "<tool_name>",
    "tool_parameters": {
        <parameters specified by the tool>
    }
}
</output>

For final output:
<output>
{
    "type": "output",
    "output": {
        <output specified by the user>
    }
}
</output>

For fatal errors, where you think you cannot continue:
<output>
{
    "type": "error",
    "reason": "<reason for the error>"
}
</output>
"""

basic_rules = """- Everything inside <output> must be valid JSON — do not include extra characters, only JSON.
- Do not include JSON or tool calls in the scratchpad; all JSON must be inside <output>.
- Do not make assumptions.
- If more info is needed, respond with the appropriate tool request.
- Only use a provided tool. Do not invent new tools. There may be no tools available.
- Do not invent results. Always use the tools if external information is required.
- You must only reply with a single tool request, an error, or with the output format specified by the user.
- Where free text is expected, do not use any format unless specified otherwise."""


def build_system_prompt(
    identity_details: str, available_tools: list[BaseTool] | None = None, extra_rules: str | None = None
) -> str:
    if available_tools:
        tools_description = "Available Tools:\n"
        for tool in available_tools:
            tools_description += f"- {tool.tool_description}\n"
    else:
        tools_description = "No tools available.\n"

    rules = basic_rules
    if extra_rules:
        rules += extra_rules

    return (
        identity_details + "\n\n" + base_system_prompt_output_description + "\n\n" + tools_description + "\n\n" + rules
    )
