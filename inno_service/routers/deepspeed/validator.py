import os

from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostDeepSpeedDefault(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostDeepSpeedDefault") -> "PostDeepSpeedDefault":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(self.train_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'train_name' does not exists",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            )

        return self


class PostDeepSpeedFile(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostDeepSpeedFile") -> "PostDeepSpeedFile":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(self.train_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'train_name' does not exists",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            )

        return self


class GetDeepSpeedPreview(BaseModel):
    ds_file_name: str

    @model_validator(mode="after")
    def check(self: "GetDeepSpeedPreview") -> "GetDeepSpeedPreview":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(self.ds_file_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ds_file_name' does not exists",
                input={"ds_file_name": self.ds_file_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            )

        return self
