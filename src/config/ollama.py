from pydantic import BaseModel


class OllamaConfig(BaseModel):
    name: str
    tag: str
    host: str
    port: int
