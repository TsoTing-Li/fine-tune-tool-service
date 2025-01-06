import re
from typing import Literal

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStartMerge(BaseModel):
    merge_name: str
    export_size: int = 5
    export_device: Literal["cpu", "auto"] = "auto"
    export_legacy_format: bool = False

    @model_validator(mode="after")
    def check(self: "PostStartMerge") -> "PostStartMerge":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.merge_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'merge_name' contain invalid characters",
                input={"merge_name": self.merge_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class PostStopMerge(BaseModel):
    merge_container: str

    @model_validator(mode="after")
    def check(self: "PostStopMerge") -> "PostStopMerge":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.merge_container):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'merge_container' contain invalid characters",
                input={"merge_container": self.merge_container},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
