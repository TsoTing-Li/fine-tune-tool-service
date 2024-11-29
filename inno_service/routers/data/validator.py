from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostData(BaseModel):
    dataset_file_name: str
    dataset_file: UploadFile

    @model_validator(mode="after")
    def check(self: "PostData") -> "PostData":
        error_handler = ResponseErrorHandler()

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
