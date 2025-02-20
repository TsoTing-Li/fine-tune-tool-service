from pydantic import BaseModel


class FineTuneToolConfig(BaseModel):
    name: str
    tag: str
