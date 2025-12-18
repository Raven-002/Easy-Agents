"""Console script for ai_cr."""

import typer
from rich.console import Console

from .cr_generator import generate_code_review
from .utils.logging_utils import dlog, set_verbosity

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
) -> None:
    """Perform code review for a given branch."""
    # Initialize verbosity level: debug > verbose > normal
    set_verbosity(2 if debug else 1 if verbose else 0)

    dlog(
        f"ai_cr {target_branch}{' --skip-print' if skip_print else ''} "
        f"--cache-dir {cache_dir}{' --save-to-file ' + save_to_file if save_to_file else ''}"
    )

    cr = generate_code_review(target_branch)
    if not skip_print:
        cr.print_comments(console)


if __name__ == "__main__":
    app()
