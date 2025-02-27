from pydantic import BaseModel


class StatusConfig(BaseModel):
    setup: str
    active: str
    finish: str
    failed: str
    stopped: str
