import re
from typing import List, Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, model_validator

from src.utils.error import ResponseErrorHandler


class PostStartEval(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    eval_name: str
    eval_type: Literal["generate", "chat"] = "generate"
    tasks: List[Literal["gsm8k"]] = ["gsm8k"]
    model_server_url: str
    num_concurrent: int = 3
    max_retries: int = 3

    @model_validator(mode="after")
    def check(self: "PostStartEval") -> "PostStartEval":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.eval_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'eval_name' contain invalid characters",
                input={"eval_name": self.eval_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            ) from None

        return self


class PostStopEval(BaseModel):
    eval_name: str

    @model_validator(mode="after")
    def check(self: "PostStopEval") -> "PostStopEval":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.eval_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'eval_name' contain invalid characters",
                input={"eval_name": self.eval_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            ) from None

        return self
