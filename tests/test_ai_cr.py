#!/usr/bin/env python
import pytest
from typer.testing import CliRunner
from ai_cr.cli import app

runner = CliRunner()

"""Tests for `ai_cr` package."""

# from ai_cr import ai_cr
def test_ai_cr_help():
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Replace this message" in result.stdout
