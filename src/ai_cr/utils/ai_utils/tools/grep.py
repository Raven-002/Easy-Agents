from .base_tool import BaseTool

tool_description = """grep: A tool to search for a pattern in a file.
- tool_name: grep
- tool_parameters: A json object with the following format:
  {
      "pattern": "<pattern to search for>",
      "path": "<path to search in>",
      "is_dir": <True if path is a directory, False otherwise>,
      "range_before": <number of lines before the match, can be 0>,
      "range_after": <number of lines after the match, can be 0>
  }
- Only call grep if you need to search for code or text in a file. Do not call tools for reasoning.
"""


class GrepTool(BaseTool):
    @property
    def tool_description(self) -> str:
        return tool_description
