from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class MaxBodySizeException(Exception):
    def __init__(self, body_len: str):
        self.body_len = body_len


class MaxbodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(body_len=self.body_len)


class PostData(BaseModel):
    dataset_file_name: str
    dataset_file: UploadFile

    @model_validator(mode="after")
    def check(self: "PostData") -> "PostData":
        error_handler = ResponseErrorHandler()

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
