import re
from typing import Union

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, model_validator

from src.utils.error import ResponseErrorHandler


class PostInferBackendStart(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str

    @model_validator(mode="after")
    def check(self: "PostInferBackendStart") -> "PostInferBackendStart":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.model_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'model_name' contain invalid characters",
                input={"model_name": self.model_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostInferBackendStop(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str

    @model_validator(mode="after")
    def check(self: "PostInferBackendStop") -> "PostInferBackendStop":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.model_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'model_name' contain invalid characters",
                input={"model_name": self.model_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class GetInferBackend(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetInferBackend") -> "GetInferBackend":
        error_handler = ResponseErrorHandler()

        if self.model_name and not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.model_name
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'model_name' contain invalid characters",
                input={"model_name": self.model_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
