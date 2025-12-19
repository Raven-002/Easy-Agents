from pydantic import BaseModel

from ..ai_utils.ai_runner import AiRunner, AiRunnerExpertise, ParsableAiExpertise
from .common_ai_utils import build_system_prompt
from .jobs import jobs


class JobRequest(BaseModel):
    required_expertise: ParsableAiExpertise
    job_type: str
    task_description: str
    expected_output_description: str

    def __post_init__(self) -> None:
        if self.job_type not in jobs.keys():
            raise ValueError(f"Job type {self.job_type} not found.")


class OrchestratedJob:
    def __init__(self, job_id: str, expertise: AiRunnerExpertise, prompt: str):
        self.__job_id = job_id
        self.__expertise = expertise
        self.__prompt = prompt


class Orchestrator:
    def __init__(self, orchestration_runner: AiRunner, code_analysis_runner: AiRunner, code_review_runner: AiRunner):
        if AiRunnerExpertise.ORCHESTRATION not in orchestration_runner.expertise:
            raise ValueError("AI runner does not support orchestration.")
        if AiRunnerExpertise.CODE_UNDERSTANDING not in code_analysis_runner.expertise:
            raise ValueError("AI runner does not support code understanding.")
        if AiRunnerExpertise.CODE_REVIEW not in code_review_runner.expertise:
            raise ValueError("AI runner does not support code review.")
        self._ai_runners = {
            AiRunnerExpertise.ORCHESTRATION: orchestration_runner,
            AiRunnerExpertise.CODE_UNDERSTANDING: code_analysis_runner,
            AiRunnerExpertise.CODE_REVIEW: code_review_runner,
        }

    def orchestrate(self, identity: str, task: str) -> list[OrchestratedJob]:
        system_prompt = build_system_prompt(
            identity_details=identity,
        )
        task += "\n\n"
        task += "Output format:\n"
        task += 'Create a json object with a list named "jobs".\n'
        task += "Each job need to be the following format:\n"
        task += "{\n"
        task += '    "job_type": "<job type/name from the available jobs>"\n'
        task += '    "extra_details": <A json object with the details the job needs>\n'
        task += "}\n"
        task += "\n\n" + self.generate_jobs_prompt_snippet()
        self._ai_runners[AiRunnerExpertise.ORCHESTRATION].run(prompt=task, system_prompt=system_prompt)
        # TODO: Generate jobs from output
        return []

    @staticmethod
    def generate_jobs_prompt_snippet() -> str:
        snippet = "Available jobs:\n"
        for job_name, job in jobs.items():
            snippet += f"- {job_name}:\n"
            snippet += f"    - description: {job.get_description()}\n"
            snippet += f"    - extra_details_required: {job.get_extra_details_description()}\n"
        snippet += "\n"

        return snippet
