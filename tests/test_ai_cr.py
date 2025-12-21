#!/usr/bin/env python
from typer.testing import CliRunner

from easy_agents.cli import app

runner = CliRunner()

"""Tests for `easy_agents` package."""


def test_ai_cr_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stderr
    assert "Perform code review for a given branch" in result.stdout
