from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from easy_agents.core import RunContext, Tool


class ListDirParameters(BaseModel):
    dir_path: str = Field(description="The directory to list.")


class ListDirEntry(BaseModel):
    name: str = Field(description="The name of the entry.")
    type: Literal["file", "dir", "symlink"] = Field(description="The type of the entry.")
    size: int = Field(description="The size of the entry in bytes.")
    created_at: str = Field(description="The creation date of the entry.")
    last_modified: str = Field(description="The last modification date of the entry.")


class ListDirResults(BaseModel):
    entries: list[ListDirEntry] = Field(description="List of entries.")
    entries_count: int = Field(description="The number of entries.")


async def _run(_ctx: RunContext, parameters: ListDirParameters) -> ListDirResults:
    path = Path(parameters.dir_path)

    if not path.exists():
        raise ValueError(f"The path {parameters.dir_path} does not exist.")
    if not path.is_dir():
        raise ValueError(f"The path {parameters.dir_path} is not a directory.")

    entries: list[ListDirEntry] = []

    for entry in path.iterdir():
        stats = entry.stat()

        entry_type: Literal["file", "dir", "symlink"]
        if entry.is_symlink():
            entry_type = "symlink"
        elif entry.is_dir():
            entry_type = "dir"
        else:
            entry_type = "file"

        entry_model = ListDirEntry(
            name=entry.name,
            type=entry_type,
            size=stats.st_size,
            created_at=datetime.fromtimestamp(stats.st_ctime).isoformat(),
            last_modified=datetime.fromtimestamp(stats.st_mtime).isoformat(),
        )
        entries.append(entry_model)

    return ListDirResults(entries=entries, entries_count=len(entries))


list_dir_tool = Tool[ListDirParameters, ListDirResults, None](
    name="list_dir_tool",
    description="List the entries in a directory. Supports symlinks and hidden files/directories.",
    run=_run,
    parameters_type=ListDirParameters,
    results_type=ListDirResults,
)
