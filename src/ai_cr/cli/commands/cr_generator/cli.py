import typer

from ai_cr.cli.app import app, console
from ai_cr.logger.logging_utils import configure_logging, dlog, status
from ai_cr.settings.settings import load_settings_from_yaml
from ai_cr.utils.git_utils.git_diff import git_diff, summerize_diff_files
from ai_cr.utils.types.code_review import CodeReview, CodeReviewGeneralComment


def generate_cr(target_branch: str) -> CodeReview:
    diff = git_diff(target_branch)
    diff_str = f"Modified Files:\n{summerize_diff_files(diff)}\n"
    diff_str += "Diff blocks:\n"
    i = 0
    for file in diff:
        for hunk in file.hunks:
            i += 1
            diff_str += f"=== {i}: {file.path_a}\n{hunk.hunk}\n\n"
    diff_context = f"<diff>\n{diff_str}\n</diff>"

    assert diff_context
    # TODO

    return CodeReview("AI-CR-BOT", [CodeReviewGeneralComment("")])


@app.command()
def cr(
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
    settings_file: str = typer.Option(".ai_cr_config.yml", envvar="AI_CR_SETTINGS_FILE", help="Path to settings file."),
) -> None:
    """Perform code review for a given branch."""
    configure_logging(level=2 if debug else 1 if verbose else 0, console=console, spinner=spinner)

    with open(settings_file) as f:
        load_settings_from_yaml(f.read())

    dlog(
        f"ai_cr {target_branch}{' --skip-print' if skip_print else ''} "
        f"--cache-dir {cache_dir}{' --save-to-file ' + save_to_file if save_to_file else ''}"
    )

    with status("Generating code review..."):
        review = generate_cr(target_branch)
    if not skip_print:
        review.print_comments(console)
