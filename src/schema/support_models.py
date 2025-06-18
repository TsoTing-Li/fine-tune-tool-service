from typing import List

from pydantic import BaseModel


class SupportModelInfo(BaseModel):
    uuid: str
    name: str
    template: str
    lora_module: List[str]
