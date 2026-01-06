from pydantic import BaseModel


class CodeDir(BaseModel):
    language: str
    description: str
    path: str


class CodeProjectContext(BaseModel):
    description: str
    project_root: str
    directories: list[CodeDir]
