import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from easy_agents.core import RunContext, ToolDepsRegistry
from easy_agents.tools import list_dir_tool


@pytest.fixture
def sample_dir() -> Iterator[str]:
    """
    Create a temporary directory with:
    - Subdirectories (nested)
    - Files
    - Symlinks (pointing to a file)
    - Hidden files (Unix-like systems)
    """
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)

        # 1. Create a subdirectory
        sub_dir = base_path / "subdir"
        sub_dir.mkdir()

        # 2. Create a file in the root
        file_path = base_path / "root_file.txt"
        file_path.write_text("Hello from root")

        # 3. Create a file in the subdirectory
        nested_file = sub_dir / "nested_file.log"
        nested_file.write_text("Log entry")

        # 4. Create a symlink (root_file.txt -> link_to_file)
        link_path = base_path / "link_to_file"
        try:
            link_path.symlink_to("root_file.txt")
        except OSError:
            # Skip symlink on Windows if permissions don't allow it
            pass

        # 5. Create a hidden file (on Unix-like systems)
        hidden_file = base_path / ".hidden_file"
        hidden_file.write_text("Hidden content")

        yield str(base_path)


@pytest.fixture
def empty_dir() -> Iterator[str]:
    """Create an empty temporary directory."""
    with TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def run_context() -> RunContext:
    """Create a basic RunContext for testing."""
    return RunContext(deps=ToolDepsRegistry.empty(), ctx=None, router=None, main_model=None)  # type: ignore


@pytest.mark.asyncio
async def test_list_dir_basic(sample_dir: str, run_context: RunContext) -> None:
    """Test listing a directory with various entry types."""
    params_dict = {"dir_path": sample_dir}
    result = await list_dir_tool.run(run_context, json.dumps(params_dict))

    # Verify counts (may vary based on symlink creation success)
    assert result.entries_count >= 3  # At minimum: subdir, root_file.txt, .hidden_file
    assert len(result.entries) == result.entries_count

    # Find specific entries
    entries_by_name = {entry.name: entry for entry in result.entries}

    # Check subdirectory
    assert "subdir" in entries_by_name
    assert entries_by_name["subdir"].type == "dir"
    assert entries_by_name["subdir"].size >= 0

    # Check root file
    assert "root_file.txt" in entries_by_name
    assert entries_by_name["root_file.txt"].type == "file"
    assert entries_by_name["root_file.txt"].size > 0

    # Check hidden file
    assert ".hidden_file" in entries_by_name
    assert entries_by_name[".hidden_file"].type == "file"

    # Verify timestamps are valid ISO format
    for entry in result.entries:
        assert entry.created_at
        assert entry.last_modified
        # Should be parsable as ISO datetime
        from datetime import datetime

        datetime.fromisoformat(entry.created_at)
        datetime.fromisoformat(entry.last_modified)


@pytest.mark.asyncio
async def test_list_dir_symlink(sample_dir: str, run_context: RunContext) -> None:
    """Test that symlinks are correctly identified."""
    params_dict = {"dir_path": sample_dir}
    result = await list_dir_tool.run(run_context, json.dumps(params_dict))

    entries_by_name = {entry.name: entry for entry in result.entries}

    # Check symlink if it was created successfully
    if "link_to_file" in entries_by_name:
        symlink_entry = entries_by_name["link_to_file"]
        assert symlink_entry.type == "symlink"
        assert symlink_entry.size >= 0
        assert symlink_entry.created_at
        assert symlink_entry.last_modified


@pytest.mark.asyncio
async def test_list_dir_empty(empty_dir: str, run_context: RunContext) -> None:
    """Test listing an empty directory."""
    params_dict = {"dir_path": empty_dir}
    result = await list_dir_tool.run(run_context, json.dumps(params_dict))

    assert result.entries_count == 0
    assert len(result.entries) == 0


@pytest.mark.asyncio
async def test_list_dir_nonexistent(run_context: RunContext) -> None:
    """Test listing a non-existent directory raises ValueError."""
    params_dict = {"dir_path": "/nonexistent/path/that/does/not/exist"}

    with pytest.raises(ValueError, match="does not exist"):
        await list_dir_tool.run(run_context, json.dumps(params_dict))


@pytest.mark.asyncio
async def test_list_dir_file_instead_of_dir(sample_dir: str, run_context: RunContext) -> None:
    """Test listing a file instead of a directory raises ValueError."""
    file_path = Path(sample_dir) / "root_file.txt"
    params_dict = {"dir_path": str(file_path)}

    with pytest.raises(ValueError, match="is not a directory"):
        await list_dir_tool.run(run_context, json.dumps(params_dict))


@pytest.mark.asyncio
async def test_list_dir_file_sizes(sample_dir: str, run_context: RunContext) -> None:
    """Test that file sizes are correctly reported."""
    params_dict = {"dir_path": sample_dir}
    result = await list_dir_tool.run(run_context, json.dumps(params_dict))

    entries_by_name = {entry.name: entry for entry in result.entries}

    # root_file.txt should have size matching its content
    root_file = entries_by_name["root_file.txt"]
    expected_size = len("Hello from root")
    assert root_file.size == expected_size

    # hidden file should also have correct size
    hidden_file = entries_by_name[".hidden_file"]
    expected_hidden_size = len("Hidden content")
    assert hidden_file.size == expected_hidden_size


@pytest.mark.asyncio
async def test_list_dir_nested_not_included(sample_dir: str, run_context: RunContext) -> None:
    """Test that nested files are not included in parent directory listing."""
    params_dict = {"dir_path": sample_dir}
    result = await list_dir_tool.run(run_context, json.dumps(params_dict))

    entries_names = {entry.name for entry in result.entries}

    # nested_file.log should NOT appear in root listing
    assert "nested_file.log" not in entries_names


@pytest.mark.asyncio
async def test_list_dir_subdirectory(sample_dir: str, run_context: RunContext) -> None:
    """Test listing a subdirectory."""
    subdir_path = Path(sample_dir) / "subdir"
    params_dict = {"dir_path": str(subdir_path)}
    result = await list_dir_tool.run(run_context, json.dumps(params_dict))

    assert result.entries_count == 1
    assert result.entries[0].name == "nested_file.log"
    assert result.entries[0].type == "file"
    assert result.entries[0].size == len("Log entry")


@pytest.mark.asyncio
async def test_list_dir_ordering(sample_dir: str, run_context: RunContext) -> None:
    """Test that entries are returned (order may vary by filesystem)."""
    params_dict = {"dir_path": sample_dir}
    result = await list_dir_tool.run(run_context, json.dumps(params_dict))

    # Just verify all entries have names
    for entry in result.entries:
        assert entry.name
        assert len(entry.name) > 0


@pytest.mark.asyncio
async def test_list_dir_special_characters() -> None:
    """Test listing directory with special characters in names."""
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)

        # Create files with special characters
        special_files = ["file with spaces.txt", "file-with-dashes.txt", "file_with_underscores.txt"]
        for filename in special_files:
            (base_path / filename).write_text("test")

        run_context = RunContext(deps=ToolDepsRegistry.empty(), ctx=None, router=None, main_model=None)  # type: ignore
        params_dict = {"dir_path": str(base_path)}
        result = await list_dir_tool.run(run_context, json.dumps(params_dict))

        entries_names = {entry.name for entry in result.entries}
        for filename in special_files:
            assert filename in entries_names
