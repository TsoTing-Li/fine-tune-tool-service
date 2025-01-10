import re
from typing import List

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStartChat(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_service: str
    chat_model_name: str
    messages: List[str]

    @model_validator(mode="after")
    def check(self: "PostStartChat") -> "PostStartChat":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\:/]+", self.model_service)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'model_service' container invalid characters",
                input={"model_service": self.model_service},
            )

        if bool(re.search(r"[^a-zA-Z0-9_\-\:]+", self.chat_model_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'chat_model_name' container invalid characters",
                input={"chat_model_name": self.chat_model_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class PostStopChat(BaseModel):
    request_id: str

    @model_validator(mode="after")
    def check(self: "PostStopChat") -> "PostStopChat":
        error_handler = ResponseErrorHandler()

        if not re.match(
            r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$",
            self.request_id,
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'request_id' contain invalid characters",
                input={"request_id": self.request_id},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self
