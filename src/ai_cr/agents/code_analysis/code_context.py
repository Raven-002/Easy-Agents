from pydantic import BaseModel


class CodeDir(BaseModel):
    language: str
    description: str
    paths: list[str]


class CodeProjectContext(BaseModel):
    description: str
    directories: list[CodeDir]
