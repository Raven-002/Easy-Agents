from pydantic import BaseModel, Field

from easy_agents.core.tool import RunContext, Tool


class _Parameters(BaseModel):
    file_path: str = Field(description="The file to read.")
    start_line: int = Field(
        0,
        description="The first line to read from. 0 means from the start. Negative values supported and are relative "
        "to the last line.",
    )
    end_line: int = Field(
        0,
        description="The last line to read. 0 means to the last line. Negative values supported and are relative to "
        "the last line.",
    )


class _Line(BaseModel):
    line_number: int = Field(description="The line number.")
    line: str = Field(description="The line content.")


class _Results(BaseModel):
    lines: list[_Line] = Field(description="List of read lines.")
    lines_count: int = Field(description="The number of lines read.")


async def _run(_ctx: RunContext, parameters: _Parameters) -> _Results:
    with open(parameters.file_path) as file:
        all_lines: list[_Line] = [_Line(line_number=i, line=content) for i, content in enumerate(file.readlines())]
    if parameters.end_line == 0:
        parameters.end_line = len(all_lines)
    lines = all_lines[parameters.start_line : parameters.end_line]
    return _Results(lines=lines, lines_count=len(lines))


read_tool = Tool[_Parameters, _Results, None](
    name="read_tool",
    description="Read the content of a file.",
    run=_run,
    parameters_type=_Parameters,
    results_type=_Results,
)
