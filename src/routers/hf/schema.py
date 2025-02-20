from pydantic import BaseModel


class PostAddToken(BaseModel):
    hf_token: str
