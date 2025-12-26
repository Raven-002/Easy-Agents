from typing import Any

from .analyze_symbol.cli import analyze_symbol
from .cr_generator.cli import cr

all_commands: list[Any] = [cr, analyze_symbol]

__all__ = ["all_commands"]
