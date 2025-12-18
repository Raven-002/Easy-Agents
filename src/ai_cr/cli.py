"""Console script for ai_cr."""

import typer
from rich.console import Console

from .cr_generator import generate_code_review
from .utils.logging_utils import configure_logging, dlog, status

app = typer.Typer()
console = Console()


@app.command()
def main(
    target_branch: str = typer.Argument(help="Target branch to generate code review for."),
    skip_print: bool = typer.Option(False, help="Skip printing generated code review."),
    cache_dir: str = typer.Option(".cr.cache", envvar="AI_CR_CACHE_DIR", help="Directory to store cache files."),
    save_to_file: str | None = typer.Option(None, help="File to save generated code review to."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", "-vv", help="Enable debug output (most verbose)"),
    spinner: bool | None = typer.Option(
        None,
        "--spinner/--no-spinner",
        help="Show a spinner while waiting for the AI model response (default: auto, only in TTY).",
    ),
) -> None:
    """Perform code review for a given branch."""
    configure_logging(level=2 if debug else 1 if verbose else 0, console=console, spinner=spinner)

    dlog(
        f"ai_cr {target_branch}{' --skip-print' if skip_print else ''} "
        f"--cache-dir {cache_dir}{' --save-to-file ' + save_to_file if save_to_file else ''}"
    )

    with status("Generating code review..."):
        cr = generate_code_review(target_branch)
    if not skip_print:
        cr.print_comments(console)


if __name__ == "__main__":
    app()
