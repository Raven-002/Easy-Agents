#!/usr/bin/env python
from typer.testing import CliRunner

from ai_cr.cli import cli_app

runner = CliRunner()

"""Tests for `ai_cr` package."""


def test_ai_cr_help() -> None:
    result = runner.invoke(cli_app, ["cr", "--help"])
    assert result.exit_code == 0, result.stderr
    assert "Perform code review for a given branch" in result.stdout
