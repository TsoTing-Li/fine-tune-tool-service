import re

from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.utils.error import ResponseErrorHandler


class PostStartQuantize(BaseModel):
    quantize_name: str

    @model_validator(mode="after")
    def check(self: "PostStartQuantize") -> "PostStartQuantize":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.quantize_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'quantize_name' contain invalid characters",
                input={"quantize_name": self.quantize_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostStopQuantize(BaseModel):
    quantize_name: str

    @model_validator(mode="after")
    def check(self: "PostStopQuantize") -> "PostStopQuantize":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.quantize_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'quantize_name' contain invalid characters",
                input={"quantize_name": self.quantize_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
