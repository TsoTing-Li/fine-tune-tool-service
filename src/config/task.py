from pydantic import BaseModel


class TaskConfig(BaseModel):
    data: str
    train: str
    merge: str
    eval: str
    chat: str
    quantize: str
    accelbrain_device: str
    deploy: str
