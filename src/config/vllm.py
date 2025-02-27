from pydantic import BaseModel


class VllmConfig(BaseModel):
    name: str
    tag: str
    host: str
    port: int
