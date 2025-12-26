from pydantic import BaseModel


class AgentDeps(BaseModel):
    agent_name: str
    agent_version: str
