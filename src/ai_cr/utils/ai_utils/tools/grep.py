import subprocess

from pydantic import BaseModel, Field

from .abstract_tool import AbstractTool


class GrepTool(AbstractTool):
    class Parameters(BaseModel):
        search_expression: str = Field(
            ..., description="The regex or text pattern to search for, using extended regular expressions (ERE)."
        )
        paths: list[str] = Field(
            ..., description="A list of file or directory paths to search within. Globs are not supported."
        )
        is_dir: bool = Field(
            ..., description="If True, performs a recursive search through all files in the provided paths."
        )
        context_before: int = Field(0, description="Number of lines of leading context to include before each match.")
        context_after: int = Field(0, description="Number of lines of trailing context to include after each match.")

    @classmethod
    def description(cls) -> str:
        return "Search for regex matches in files or directories using grep."

    @classmethod
    def _run(cls, params: Parameters) -> str:
        """Run grep with the provided parameters."""
        flags = ["-nH"]  # line numbers + filename

        if params.is_dir:
            flags.append("-r")
        if params.context_before > 0:
            flags.extend(["-B", str(params.context_before)])
        if params.context_after > 0:
            flags.extend(["-A", str(params.context_after)])

        # Use extended regex
        command = ["grep"] + flags + ["-E", params.search_expression] + params.paths

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout
            elif result.returncode == 1:
                return "No matches found."
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"System Error: {str(e)}"
