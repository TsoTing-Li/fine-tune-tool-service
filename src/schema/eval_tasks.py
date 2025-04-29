from pydantic import BaseModel


class EvalTaskInfo(BaseModel):
    uuid: str
    name: str
    tool_input: str
    group: str
