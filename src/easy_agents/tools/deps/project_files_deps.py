from pydantic import BaseModel, Field

from easy_agents.core.tool import ToolDependency


class ProjectFilesDeps(BaseModel):
    project_root: str = Field(description="The root of the project.")


project_files_deps_type = ToolDependency[ProjectFilesDeps](key="project_files", value_type=ProjectFilesDeps)
