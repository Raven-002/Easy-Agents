from pydantic import BaseModel, Field


class ProjectFilesDeps(BaseModel):
    project_root: str = Field(description="The root of the project.")
