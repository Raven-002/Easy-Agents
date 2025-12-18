from __future__ import annotations

from rich.console import Console

# Verbosity levels
_LEVEL_NORMAL = 0
_LEVEL_VERBOSE = 1
_LEVEL_DEBUG = 2

_verbosity_level: int = _LEVEL_NORMAL
_console: Console | None = None


def _get_console() -> Console:
    global _console
    if _console is None:
        _console = Console()
    return _console


def set_verbosity(level: int) -> None:
    """Set global verbosity level.

    Levels:
    - 0: normal (no extra logs)
    - 1: verbose
    - 2: debug (most verbose)
    """
    global _verbosity_level
    _verbosity_level = max(_LEVEL_NORMAL, min(level, _LEVEL_DEBUG))


def is_verbose() -> bool:
    return _verbosity_level >= _LEVEL_VERBOSE


def is_debug() -> bool:
    return _verbosity_level >= _LEVEL_DEBUG


def vlog(message: str) -> None:
    if is_verbose():
        _get_console().log(message)


def dlog(message: str) -> None:
    if is_debug():
        _get_console().log(message)
