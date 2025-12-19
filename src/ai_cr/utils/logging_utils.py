from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console

# Verbosity levels
_LEVEL_NORMAL = 0
_LEVEL_VERBOSE = 1
_LEVEL_DEBUG = 2

_verbosity_level: int = _LEVEL_NORMAL
_console: Console | None = None
_spinner_enabled: bool | None = None  # None means auto (TTY-only)


def _get_console() -> Console:
    global _console
    if _console is None:
        _console = Console()
    return _console


def configure_logging(level: int, console: Console | None = None, spinner: bool | None = None) -> None:
    """Configure global logging behavior.

    - level: Verbosity level (0-2).
    - console: Rich Console instance to use for logging and spinners.
    - spinner: True to force-enable spinner, False to disable, None for auto (enabled only in TTY).
    """
    global _verbosity_level, _console, _spinner_enabled
    _verbosity_level = max(_LEVEL_NORMAL, min(level, _LEVEL_DEBUG))
    if console is not None:
        _console = console
    _spinner_enabled = spinner


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


@contextmanager
def status(message: str) -> Iterator[None]:
    """Show a spinner/status while inside the context, if enabled.

    Spinner enablement precedence:
    1) Explicit `spinner` passed to `configure_logging`.
    2) Auto: enabled if the console is a TTY (interactive terminal).
    """
    console = _get_console()
    # Determine enablement
    if _spinner_enabled is None:
        show = console.is_terminal
    else:
        show = bool(_spinner_enabled)

    if show:
        with console.status(message, spinner="dots"):
            yield
    else:
        _get_console().print(message)
        yield
