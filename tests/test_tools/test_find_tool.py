import json
from pathlib import Path

import pytest

from easy_agents.core import RunContext, ToolDepsRegistry
from easy_agents.tools.deps.project_files_deps import ProjectFilesDeps, project_files_deps_type
from easy_agents.tools.find_tool import find_tool


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a temporary project root with some files."""
    (tmp_path / "file1.txt").write_text("Hello world\nThis is a test file.\nRegex is cool.", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file2.txt").write_text("Another file in subdir.\nHello again.", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"Some binary data\x00with null byte")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("This file should be ignored.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def run_context(project_root: Path) -> RunContext:
    """Create a RunContext with ProjectFilesDeps."""
    deps = ToolDepsRegistry.from_map({project_files_deps_type: ProjectFilesDeps(project_root=str(project_root))})
    return RunContext(deps=deps, ctx=None, router=None, main_model=None)  # type: ignore


@pytest.mark.asyncio
async def test_find_single_file(run_context: RunContext, project_root: Path) -> None:
    """Test searching in a single file."""
    params = {
        "search_expression": "test",
        "paths": ["file1.txt"],
        "is_dir": False,
        "context_before": 0,
        "context_after": 0,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    assert result.matches_count == 1
    assert result.matches[0].path == str(project_root / "file1.txt")
    assert result.matches[0].line_number == 2
    assert "test file" in result.matches[0].line
    assert result.matches[0].match == "test"


@pytest.mark.asyncio
async def test_find_recursive(run_context: RunContext, project_root: Path) -> None:
    """Test recursive search in a directory."""
    params = {
        "search_expression": "Hello",
        "paths": ["."],
        "is_dir": True,
        "context_before": 0,
        "context_after": 0,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    # Hello world in file1.txt, Hello again in subdir/file2.txt
    assert result.matches_count == 2
    paths = {m.path for m in result.matches}
    assert str(project_root / "file1.txt") in paths
    assert str(project_root / "subdir" / "file2.txt") in paths


@pytest.mark.asyncio
async def test_find_with_context(run_context: RunContext, project_root: Path) -> None:
    """Test searching with context lines."""
    params = {
        "search_expression": "test",
        "paths": ["file1.txt"],
        "is_dir": False,
        "context_before": 1,
        "context_after": 1,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    assert result.matches_count == 1
    # Context before: Hello world, Match: This is a test file., Context after: Regex is cool.
    expected_line = "Hello world\nThis is a test file.\nRegex is cool."
    assert result.matches[0].line == expected_line


@pytest.mark.asyncio
async def test_find_ignore_gitignore(run_context: RunContext, project_root: Path) -> None:
    """Test that .gitignore is respected."""
    params = {
        "search_expression": "This file should be ignored",
        "paths": ["."],
        "is_dir": True,
        "context_before": 0,
        "context_after": 0,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    # ignored.txt contains "This file should be ignored", but it should be ignored by .gitignore
    assert result.matches_count == 0


@pytest.mark.asyncio
async def test_find_skip_binary(run_context: RunContext, project_root: Path) -> None:
    """Test that binary files are skipped."""
    params = {
        "search_expression": "binary",
        "paths": ["."],
        "is_dir": True,
        "context_before": 0,
        "context_after": 0,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    # binary.bin contains "binary", but it's a binary file
    assert result.matches_count == 0


@pytest.mark.asyncio
async def test_find_invalid_regex(run_context: RunContext) -> None:
    """Test handling of invalid regex patterns."""
    params = {
        "search_expression": "[unclosed bracket",
        "paths": ["file1.txt"],
        "is_dir": False,
        "context_before": 0,
        "context_after": 0,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    assert result.matches_count == 0
    assert result.matches == []


@pytest.mark.asyncio
async def test_find_nonexistent_path(run_context: RunContext) -> None:
    """Test searching in a non-existent path."""
    params = {
        "search_expression": "hello",
        "paths": ["nonexistent.txt"],
        "is_dir": False,
        "context_before": 0,
        "context_after": 0,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    assert result.matches_count == 0


@pytest.mark.asyncio
async def test_find_multiple_paths(run_context: RunContext, project_root: Path) -> None:
    """Test searching in multiple paths."""
    params = {
        "search_expression": "file",
        "paths": ["file1.txt", "subdir/file2.txt"],
        "is_dir": False,
        "context_before": 0,
        "context_after": 0,
    }
    result = await find_tool.run(run_context, json.dumps(params))
    # "test file" in file1.txt, "Another file" in subdir/file2.txt
    assert result.matches_count == 2
