from dataclasses import dataclass

import pydantic

from easy_agents.easy_agents import AiTask, Context, GrepTool
from easy_agents.settings.settings import get_settings
from easy_agents.templates.templates_loader import render_template
from easy_agents.utils.git_utils.git_diff import git_diff, summerize_diff_files
from easy_agents.utils.types.code_review import CodeReview, CodeReviewGeneralComment


@dataclass
class OrchestratorResponse:
    status: str = pydantic.Field(description="plain text for status message.")


orchestrator = AiTask(
    prompt="Orchestrate a code review on the provided diff.", output_schema=OrchestratorResponse, tools=[GrepTool]
)


@dataclass
class FinderResponse:
    output: str = pydantic.Field(description="Markdown text for output message.")


test_task = AiTask(
    system_prompt=render_template("system_prompt.j2"),
    prompt="Find how many time the word 'world' and any similar in meaning word exists in the tests dir, using the"
    " path 'tests'. Think about which words are similar to world.",
    output_schema=FinderResponse,
    tools=[GrepTool],
)


def generate_code_review(target_branch: str) -> CodeReview:
    runner = get_settings().create_orchestrator_runner()

    diff = git_diff(target_branch)
    diff_str = f"Modified Files:\n{summerize_diff_files(diff)}\n"
    diff_str += "Diff blocks:\n"
    i = 0
    for file in diff:
        for hunk in file.hunks:
            i += 1
            diff_str += f"=== {i}: {file.path_a}\n{hunk.hunk}\n\n"
    diff_context = f"<diff>\n{diff_str}\n</diff>"
    context = Context()
    context.add_user_message(diff_context)

    # orchestrator.run(runner, context)
    test_task.run(runner, Context())

    return CodeReview("AI-CR-BOT", [CodeReviewGeneralComment("")])
