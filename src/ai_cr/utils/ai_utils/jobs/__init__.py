from .base_job import BaseJob
from .diff_analysis import DiffAnalysisJob

jobs: dict[str, type[BaseJob]] = {"diff_analysis_job": DiffAnalysisJob}
