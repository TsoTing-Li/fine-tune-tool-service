from pydantic import BaseModel


class HwInfoConfig(BaseModel):
    name: str
    tag: str
    container_name: str
