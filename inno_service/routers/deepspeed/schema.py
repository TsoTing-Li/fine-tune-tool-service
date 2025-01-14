import re
from typing import Literal

from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostDeepSpeedDefault(BaseModel):
    stage: Literal[2, 3]
    enable_offload: bool = False
    offload_device: Literal["cpu", "nvme", None] = None

    @model_validator(mode="after")
    def check(self: "PostDeepSpeedDefault") -> "PostDeepSpeedDefault":
        error_handler = ResponseErrorHandler()

        if self.enable_offload and not self.offload_device:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="must select 'offload_device' when 'enabled_offload'",
                input={
                    "enabled_offload": self.enable_offload,
                    "offload_device": self.offload_device,
                },
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PostDeepSpeedFile(BaseModel):
    ds_file: UploadFile

    @model_validator(mode="after")
    def check(self: "PostDeepSpeedFile") -> "PostDeepSpeedFile":
        error_handler = ResponseErrorHandler()

        if self.ds_file.content_type != "application/json":
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'content_type' must be 'application/json'",
                input={"ds_file": f"{self.ds_file.content_type}"},
            )

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.ds_file.filename)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ds_file filename' contain invalid characters",
                input={"ds_file filename": self.ds_file.filename},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class GetDeepSpeedPreview(BaseModel):
    ds_file_name: str

    @model_validator(mode="after")
    def check(self: "GetDeepSpeedPreview") -> "GetDeepSpeedPreview":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.ds_file_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ds_file_name' contain invalid characters",
                input={"dataset_src": self.ds_file_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class DelDeepSpeed(BaseModel):
    ds_file_name: str

    @model_validator(mode="after")
    def check(self: "DelDeepSpeed") -> "DelDeepSpeed":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.ds_file_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ds_file_name' contain invalid characters",
                input={"dataset_src": self.ds_file_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
