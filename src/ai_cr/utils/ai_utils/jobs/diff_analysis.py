from ...common import RecursiveStrDict
from ..ai_runner import AiRunner, AiRunnerExpertise
from ..common_ai_utils import build_system_prompt
from ..tools.grep import GrepTool
from .base_job import BaseJob, CommonJobParams

job_description = (
    "Analyze diff blocks to understand their changes with context, importance and size effects. Each "
    "diff job should be on as small amount of diff as possible. Give only diff blocks relevant to each other in the "
    "same job. For unrelated diff blocks, create multiple jobs."
)

extra_details_description = """A json object with a list of diff block numbers, with the following format:
{
    "diff_block_numbers": [
    ]
}
"""

identity_details = """You are an autonomous code analyzer."""

prompt = """Analyze the following diff blocks:
{diff_blocks}
"""


class DiffAnalysisJob(BaseJob):
    @classmethod
    def get_description(cls) -> str:
        return job_description

    @classmethod
    def get_expertise(cls) -> AiRunnerExpertise:
        return AiRunnerExpertise.CODE_UNDERSTANDING

    @classmethod
    def get_extra_details_description(cls) -> str | None:
        return extra_details_description

    def run(self, ai_runner: AiRunner, common_params: CommonJobParams, extra_details: RecursiveStrDict) -> str:
        tools = [GrepTool()]
        system_prompt = build_system_prompt(identity_details=identity_details, available_tools=tools)
        output = ai_runner.run(prompt=prompt.format(diff_blocks=str(common_params.diff)), system_prompt=system_prompt)
        return output
