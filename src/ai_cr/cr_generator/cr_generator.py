from ..code_review import CodeReview, CodeReviewGeneralComment
from ..settings.settings import get_settings
from ..utils.ai_utils.orchastrator import Orchestrator
from ..utils.git_utils.git_diff import git_diff


def generate_code_review(target_branch: str) -> CodeReview:
    settings = get_settings()

    orchestrator = Orchestrator(
        orchestration_runner=settings.create_orchestrator_runner(),
        code_analysis_runner=settings.create_code_analysis_runner(),
        code_review_runner=settings.create_code_review_runner(),
    )

    identity_details = (
        "You are an orchestrator of a code review. You decide what future jobs are needed to perform a code review."
    )

    diff = git_diff(target_branch)
    diff_str = "Modified Files:\n"
    for file in diff:
        if file.change_type == "D":
            diff_str += f"- {file.path_b} Deleted\n"
        elif file.change_type == "A":
            diff_str += f"- {file.path_a} Added\n"
        else:
            if file.permissions_a != file.permissions_b:
                diff_str += f"- {file.path_b}:{file.permissions_b} --> {file.path_a}:{file.permissions_a}\n"
            elif file.path_a != file.path_b:
                diff_str += f"- {file.path_b} --> {file.path_a}\n"
            else:
                diff_str += f"- {file.path_a} Modified\n"
    diff_str += "\n"
    diff_str += "Diff blocks:\n"
    i = 0
    for file in diff:
        for hunk in file.hunks:
            i += 1
            diff_str += f"=== {i}: {file.path_a}\n{hunk.hunk}\n\n"

    task = "Orchestrate a code review for the following diff:\n"
    task += "<diff>\n"
    task += diff_str
    task += "</diff>\n"

    jobs = orchestrator.orchestrate(identity=identity_details, task=task)

    return CodeReview("AI-CR-BOT", [CodeReviewGeneralComment(str(jobs))])
