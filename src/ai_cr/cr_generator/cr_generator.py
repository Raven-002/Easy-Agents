from ..code_review import CodeReview, CodeReviewGeneralThread, CodeReviewGeneralComment
from ..utils.ai_runner import create_ai_runner

def generate_code_review(target_branch: str) -> CodeReview:
    runner = create_ai_runner("local_ollama", model="qwen3:8b")
    cr = runner.run(f"Review {target_branch}")
    return CodeReview("AI-CR-BOT", [CodeReviewGeneralComment(cr)])
    # raise NotImplementedError
