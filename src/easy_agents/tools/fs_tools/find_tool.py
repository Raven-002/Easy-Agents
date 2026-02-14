import re
from pathlib import Path

import pathspec
from pydantic import BaseModel, Field

from easy_agents.core import RunContext, Tool

from .deps.project_files_deps import ProjectFilesDeps, project_files_deps_type


class FindParameters(BaseModel):
    search_expression: str = Field(description="The regex pattern to search for, using Python's re syntax.")
    paths: list[str] = Field(description="A list of file or directory paths to search within. Globs are not supported.")
    is_dir: bool = Field(description="If True, performs a recursive search through all files in the provided paths.")
    context_before: int = Field(0, description="Number of lines of leading context to include before each match.")
    context_after: int = Field(0, description="Number of lines of trailing context to include after each match.")


class FindMatch(BaseModel):
    """A match"""

    path: str = Field(description="The path of the match.")
    line_number: int = Field(description="The line number of the match.")
    line: str = Field(description="The line content with the match.")
    match: str = Field(description="The match.")


class FindResults(BaseModel):
    matches: list[FindMatch] = Field(description="List of matches found.")
    matches_count: int = Field(description="The number of matches found.")


def is_binary(file_path: Path, chunk_size: int = 1024) -> bool:
    """
    Check if a file is binary by looking for null bytes in the first chunk.
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
            return b"\0" in chunk
    except OSError:
        return True  # Treat unreadable files as binary/skip


async def _run(_ctx: RunContext, deps: ProjectFilesDeps, parameters: FindParameters) -> FindResults:
    """Search for text patterns in files using regular expressions."""
    matches: list[FindMatch] = []

    try:
        # Compile the regex pattern
        pattern = re.compile(parameters.search_expression)
    except re.error:
        # Return empty results if regex is invalid
        return FindResults(matches=[], matches_count=0)

    # Collect all files to search
    files_to_search: list[Path] = []
    project_root = Path(deps.project_root)

    for path_str in parameters.paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = project_root / path

        if not path.exists():
            continue

        if parameters.is_dir and path.is_dir():
            for f in path.rglob("*"):
                if ".git" in f.parts:
                    continue

                if not f.is_file():
                    continue

                # Check all .gitignore files from the file up to the project root
                is_ignored = False
                current = f
                while current.parent != current:
                    gitignore = current.parent / ".gitignore"
                    if gitignore.exists():
                        try:
                            with open(gitignore, encoding="utf-8") as file:
                                spec = pathspec.PathSpec.from_lines("gitignore", file)
                                if spec.match_file(str(f.relative_to(current.parent))):
                                    is_ignored = True
                                    break
                        except Exception:
                            pass
                    if current.parent == project_root:
                        break
                    current = current.parent

                if is_ignored:
                    continue

                if is_binary(f):
                    continue

                files_to_search.append(f)
        elif path.is_file():
            files_to_search.append(path)

    # Search through each file
    for file_path in files_to_search:
        try:
            # Try to read as a text file
            with open(file_path, encoding="utf-8", errors="strict") as file:
                lines = file.readlines()

            # Search each line
            for line_num, line in enumerate(lines, start=1):
                match_obj = pattern.search(line)
                if match_obj:
                    # Calculate context lines
                    start_idx = max(0, line_num - 1 - parameters.context_before)
                    end_idx = min(len(lines), line_num + parameters.context_after)

                    # Build context
                    context_lines: list[str] = []
                    for i in range(start_idx, end_idx):
                        context_lines.append(lines[i].rstrip("\n"))

                    # Join context with the matched line
                    full_line = (
                        "\n".join(context_lines)
                        if parameters.context_before > 0 or parameters.context_after > 0
                        else line.rstrip("\n")
                    )

                    matches.append(
                        FindMatch(path=str(file_path), line_number=line_num, line=full_line, match=match_obj.group())
                    )
        except (OSError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

    return FindResults(matches=matches, matches_count=len(matches))


find_tool = Tool[FindParameters, FindResults, ProjectFilesDeps](
    name="find_tool",
    description=(
        "Search for text patterns in files using regular expressions. Can search single files or recursively "
        "through directories."
    ),
    run=_run,
    deps_type=project_files_deps_type,
    parameters_type=FindParameters,
    results_type=FindResults,
)
