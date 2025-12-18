#!/usr/bin/env python
from typer.testing import CliRunner

from ai_cr.cli import app

runner = CliRunner()

"""Tests for `ai_cr` package."""


# from ai_cr import ai_cr
def test_ai_cr_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stderr
    assert "Perform code review for a given branch" in result.stdout
