"""Console script for ai_cr."""

import typer
from rich.console import Console

from .cr_generator import generate_code_review

app = typer.Typer()
console = Console()


@app.command()
def main(target_branch: str = typer.Argument(help="Target branch to generate code review for."),
         skip_print: bool = typer.Option(False, help="Skip printing generated code review."),
         cache_dir: str = typer.Option(".cr.cache", envvar="AI_CR_CACHE_DIR", help="Directory to store cache files."),
         save_to_file: str | None = typer.Option(None, help="File to save generated code review to."),
         verbose: bool = typer.Option(False, "--verbose", "-v", help="Print verbose output")) -> None:
    """Perform code review for a given branch."""
    if verbose:
        console.log(
            f"ai_cr {target_branch}{' --skip-print' if skip_print else ''} "
            f"--cache-dir {cache_dir}{' --save-to-file ' + save_to_file if save_to_file else ''}")

    cr = generate_code_review(target_branch)
    if not skip_print:
        cr.print_comments(console)


if __name__ == "__main__":
    app()
