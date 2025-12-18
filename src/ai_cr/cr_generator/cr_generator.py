from ..code_review import CodeReview, CodeReviewGeneralComment
from ..utils.ai_utils.ai_runner import create_ai_runner
from ..utils.git_utils.git_diff import git_diff


def generate_code_review(target_branch: str) -> CodeReview:
    diff = git_diff(target_branch)
    diff_str = str(diff)
    prompt = f"Review the following diff from current state to {target_branch}:\n<diff>\n{diff_str}\n</diff>"
    runner = create_ai_runner("local_ollama", model="qwen3:8b")
    cr = runner.run(prompt)
    return CodeReview("AI-CR-BOT", [CodeReviewGeneralComment(cr)])
