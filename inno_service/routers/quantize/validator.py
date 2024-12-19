import os

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostQuantize(BaseModel):
    checkpoint_path: str

    @model_validator(mode="after")
    def check(self: "PostQuantize") -> "PostQuantize":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(self.checkpoint_path):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="checkpoint path does not exists",
                input={"checkpoint_path": self.checkpoint_path},
            )
        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return
