from pydantic import BaseModel


class QuantizeServiceConfig(BaseModel):
    name: str
    tag: str
    host: str
    port: int
    container_name: str
    gguf_tag: str
