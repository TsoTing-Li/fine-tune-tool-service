import re

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStartOllama(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str

    @model_validator(mode="after")
    def check(self: "PostStartOllama") -> "PostStartOllama":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.model_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'model_name' contain invalid characters",
                input={"model_name": self.model_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostStopOllama(BaseModel):
    ollama_container: str

    @model_validator(mode="after")
    def check(self: "PostStopOllama") -> "PostStopOllama":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.ollama_container):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ollama_container' contain invalid characters",
                input={"ollama_container": self.ollama_container},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
