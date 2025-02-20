from pydantic import BaseModel


class RedisConfig(BaseModel):
    name: str
    username: str
    tag: str
    host: str
    port: int
    password: str
    container_name: str
