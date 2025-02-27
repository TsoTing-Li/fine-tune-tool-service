from pydantic import BaseModel


class LmEvalConfig(BaseModel):
    name: str
    tag: str
