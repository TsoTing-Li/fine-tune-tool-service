from pydantic import BaseModel


class MainServiceConfig(BaseModel):
    name: str
    tag: str
    container_name: str
    host: str
    port: int
