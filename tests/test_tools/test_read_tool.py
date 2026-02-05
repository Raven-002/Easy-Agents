import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from pydantic import ValidationError

from easy_agents.core import RunContext, ToolDepsRegistry
from easy_agents.tools.read_tool import read_tool


@pytest.fixture
def sample_file() -> Iterator[str]:
    """Create a temporary file with sample content."""
    with NamedTemporaryFile(mode="w", suffix=".txt") as f:
        f.write("Line 0\n")
        f.write("Line 1\n")
        f.write("Line 2\n")
        f.write("Line 3\n")
        f.write("Line 4\n")
        f.flush()
        yield f.name


@pytest.fixture
def empty_file() -> Iterator[str]:
    """Create an empty temporary file."""
    with NamedTemporaryFile(mode="w", suffix=".txt") as f:
        yield f.name


@pytest.fixture
def single_line_file() -> Iterator[str]:
    """Create a file with a single line."""
    with NamedTemporaryFile(mode="w", suffix=".txt") as f:
        f.write("Only line\n")
        f.flush()
        yield f.name


@pytest.fixture
def run_context() -> RunContext:
    """Create a basic RunContext for testing."""
    return RunContext(deps=ToolDepsRegistry.empty(), ctx=None, router=None, main_model=None)  # type: ignore


@pytest.mark.asyncio
async def test_read_entire_file(sample_file: str, run_context: RunContext) -> None:
    """Test reading an entire file (default behavior)."""
    params_dict = {"file_path": sample_file, "start_line": 0, "end_line": 0}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 5
    assert len(result.lines) == 5
    assert result.lines[0].line_number == 0
    assert result.lines[0].line == "Line 0\n"
    assert result.lines[4].line_number == 4
    assert result.lines[4].line == "Line 4\n"


@pytest.mark.asyncio
async def test_read_with_start_line(sample_file: str, run_context: RunContext) -> None:
    """Test reading from a specific start line."""
    params_dict = {"file_path": sample_file, "start_line": 2, "end_line": 0}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 3
    assert result.lines[0].line_number == 2
    assert result.lines[0].line == "Line 2\n"
    assert result.lines[-1].line_number == 4


@pytest.mark.asyncio
async def test_read_with_end_line(sample_file: str, run_context: RunContext) -> None:
    """Test reading up to a specific end line."""
    params_dict = {"file_path": sample_file, "start_line": 0, "end_line": 3}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 3
    assert result.lines[0].line_number == 0
    assert result.lines[0].line == "Line 0\n"
    assert result.lines[-1].line_number == 2
    assert result.lines[-1].line == "Line 2\n"


@pytest.mark.asyncio
async def test_read_range(sample_file: str, run_context: RunContext) -> None:
    """Test reading a specific range of lines."""
    params_dict = {"file_path": sample_file, "start_line": 1, "end_line": 4}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 3
    assert result.lines[0].line_number == 1
    assert result.lines[0].line == "Line 1\n"
    assert result.lines[-1].line_number == 3
    assert result.lines[-1].line == "Line 3\n"


@pytest.mark.asyncio
async def test_read_negative_start_line(sample_file: str, run_context: RunContext) -> None:
    """Test reading with negative start line (from end)."""
    params_dict = {"file_path": sample_file, "start_line": -2, "end_line": 0}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 2
    assert result.lines[0].line_number == 3
    assert result.lines[0].line == "Line 3\n"
    assert result.lines[-1].line_number == 4
    assert result.lines[-1].line == "Line 4\n"


@pytest.mark.asyncio
async def test_read_negative_end_line(sample_file: str, run_context: RunContext) -> None:
    """Test reading with negative end line."""
    params_dict = {"file_path": sample_file, "start_line": 0, "end_line": -2}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 3
    assert result.lines[0].line_number == 0
    assert result.lines[0].line == "Line 0\n"
    assert result.lines[-1].line_number == 2
    assert result.lines[-1].line == "Line 2\n"


@pytest.mark.asyncio
async def test_read_empty_file(empty_file: str, run_context: RunContext) -> None:
    """Test reading an empty file."""
    params_dict = {"file_path": empty_file, "start_line": 0, "end_line": 0}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 0
    assert len(result.lines) == 0


@pytest.mark.asyncio
async def test_read_single_line_file(single_line_file: str, run_context: RunContext) -> None:
    """Test reading a file with only one line."""
    params_dict = {"file_path": single_line_file, "start_line": 0, "end_line": 0}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    assert result.lines_count == 1
    assert result.lines[0].line_number == 0
    assert result.lines[0].line == "Only line\n"


@pytest.mark.asyncio
async def test_read_nonexistent_file(run_context: RunContext) -> None:
    """Test reading a file that doesn't exist."""
    params_dict = {"file_path": "/nonexistent/path/to/file.txt", "start_line": 0, "end_line": 0}

    with pytest.raises(FileNotFoundError):
        await read_tool.run(run_context, json.dumps(params_dict))


@pytest.mark.asyncio
async def test_invalid_range_start_greater_than_end(sample_file: str, run_context: RunContext) -> None:
    """Test with start_line greater than end_line."""
    params_dict = {"file_path": sample_file, "start_line": 4, "end_line": 2}

    result = await read_tool.run(run_context, json.dumps(params_dict))

    # Python slicing handles this gracefully by returning empty list
    assert result.lines_count == 0
    assert len(result.lines) == 0


def test_tool_metadata():
    """Test that the tool has correct metadata."""
    assert read_tool.name == "read_tool"
    assert read_tool.description == "Read the content of a file."

    schema = read_tool.get_json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_tool"
    assert "file_path" in schema["function"]["parameters"]["properties"]
    assert "start_line" in schema["function"]["parameters"]["properties"]
    assert "end_line" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_parameters_validation_missing_field(run_context: RunContext) -> None:
    """Test that missing required fields raise validation error."""
    params_dict = {
        "start_line": 0,
        "end_line": 0,
        # Missing file_path
    }

    with pytest.raises(ValidationError):  # Pydantic validation error
        await read_tool.run(run_context, json.dumps(params_dict))


@pytest.mark.asyncio
async def test_parameters_validation_wrong_type(run_context: RunContext) -> None:
    """Test that wrong parameter types raise validation error."""
    params_dict = {
        "file_path": 123,  # Should be string
        "start_line": 0,
        "end_line": 0,
    }

    with pytest.raises(ValidationError):  # Pydantic validation error
        await read_tool.run(run_context, json.dumps(params_dict))


@pytest.mark.asyncio
async def test_read_file_without_newline_at_end(run_context: RunContext) -> None:
    """Test reading a file that doesn't end with a newline."""
    with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Line without newline")
        temp_path = f.name

    try:
        params_dict = {"file_path": temp_path, "start_line": 0, "end_line": 0}

        result = await read_tool.run(run_context, json.dumps(params_dict))

        assert result.lines_count == 1
        assert result.lines[0].line == "Line without newline"
    finally:
        Path(temp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_read_unicode_content(run_context: RunContext) -> None:
    """Test reading a file with unicode characters."""
    with NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
        f.write("Hello 世界\n")
        f.write("Emoji: 😀\n")
        temp_path = f.name

    try:
        params_dict = {"file_path": temp_path, "start_line": 0, "end_line": 0}

        result = await read_tool.run(run_context, json.dumps(params_dict))

        assert result.lines_count == 2
        assert "世界" in result.lines[0].line
        assert "😀" in result.lines[1].line
    finally:
        Path(temp_path).unlink(missing_ok=True)
