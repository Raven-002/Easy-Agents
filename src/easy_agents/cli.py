#!/usr/bin/env python
"""
Console script for easy_agents.
"""

# Import commands from all agents.
from .agents.cr_agent import cli as cr_cli
from .app import app

# Reference all commands in __all__.
__all__ = ["app", "cr_cli"]

if __name__ == "__main__":
    app()
