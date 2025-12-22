"""Console script for ai_cr."""

from ai_cr import __version__

from .app import app as cli_app
from .commands import all_commands

__all__ = ["cli_app"]

# Assert all commands is a non-empty list
assert all_commands


@cli_app.command()
def version() -> None:
    print(__version__)


if __name__ == "__main__":
    cli_app()
