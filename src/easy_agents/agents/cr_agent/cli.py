import typer
from rich.console import Console

from easy_agents.agents.cr_agent import generate_code_review
from easy_agents.app import app
from easy_agents.settings.settings import load_settings_from_yaml
from easy_agents.utils.logging_utils import configure_logging, dlog, status

console = Console()


@app.command()
def perform_cr(
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
    settings_file: str = typer.Option(
        ".easy_agents_config.yml", envvar="AI_CR_SETTINGS_FILE", help="Path to settings file."
    ),
) -> None:
    """Perform code review for a given branch."""
    configure_logging(level=2 if debug else 1 if verbose else 0, console=console, spinner=spinner)

    with open(settings_file) as f:
        load_settings_from_yaml(f.read())

    dlog(
        f"easy_agents {target_branch}{' --skip-print' if skip_print else ''} "
        f"--cache-dir {cache_dir}{' --save-to-file ' + save_to_file if save_to_file else ''}"
    )

    with status("Generating code review..."):
        cr = generate_code_review(target_branch)
    if not skip_print:
        cr.print_comments(console)
