import re

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostQuantize(BaseModel):
    quantize_name: str

    @model_validator(mode="after")
    def check(self: "PostQuantize") -> "PostQuantize":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.quantize_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'quantize_name' contain invalid characters",
                input={"quantize_name": self.quantize_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
