"""Console script for ai_cr."""

import typer
from rich.console import Console

from ai_cr import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for ai_cr."""
    console.print("Replace this message by putting your code into ai_cr.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
