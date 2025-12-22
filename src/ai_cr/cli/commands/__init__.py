from typing import Any

from .cr_generator.cli import cr

all_commands: list[Any] = [cr]

__all__ = ["all_commands"]
