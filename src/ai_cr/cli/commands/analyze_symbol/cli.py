import asyncio
from typing import Any

import typer
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from ai_cr.agents.code_analysis.code_context import CodeDir, CodeProjectContext
from ai_cr.agents.code_analysis.symbol_analyzer import CodeAnalysisResults, create_symbol_analyzer
from ai_cr.cli.app import app, console
from ai_cr.logger.logging_utils import configure_logging
from ai_cr.settings.settings import get_settings, load_settings_from_yaml
from ai_cr.utils.agents_runner import run_agent


def print_analysis_results(results: CodeAnalysisResults) -> None:
    # 1. Metadata Table (Type and Files)
    meta_table = Table(show_header=False, box=None, padding=(0, 1))
    meta_table.add_column("Key", style="bold cyan")
    meta_table.add_column("Value")
    meta_table.add_row("Type:", results.type_description)
    meta_table.add_row("Files:", ", ".join(results.files_it_exists_in))

    # 2. Group all elements to go inside the Panel
    panel_content = Group(
        meta_table,
        Rule(style="dim"),
        "[bold yellow]Description[/bold yellow]",
        Markdown(results.description),
        "\n[bold yellow]Usage[/bold yellow]",
        Markdown(results.usage),
    )

    # 3. Print the Panel
    console.print(
        Panel(
            panel_content,
            title=f"[bold green]Analysis: {results.symbol_name}[/bold green]",
            title_align="left",
            border_style="bright_blue",
            expand=False,
            padding=(1, 2),
        )
    )


@app.command()
def analyze_symbol(
    symbol: str = typer.Argument(help="Symbol name to analyze."),
    language: str = typer.Option("any", help="Language of the symbol"),
    path: str = typer.Option(default=".", help="Paths to look at"),
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

    project_context = CodeProjectContext(
        description="",
        project_root=".",
        directories=[
            CodeDir(
                language=language,
                description="",
                path=path,
            )
        ],
    )
    analyzer_agent = create_symbol_analyzer(get_settings().code_analysis_model)

    results: CodeAnalysisResults | Any = asyncio.run(
        run_agent(agent=analyzer_agent, prompt=f"Analyze the symbol `{symbol}`", deps=project_context)
    )

    assert isinstance(results, CodeAnalysisResults)
    print_analysis_results(results)
