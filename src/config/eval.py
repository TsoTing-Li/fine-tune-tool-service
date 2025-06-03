from pydantic import BaseModel


class EvalConfig(BaseModel):
    name: str
    tag: str
