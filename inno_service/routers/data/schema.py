import re
from typing import Literal, Union

from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, model_validator

from inno_service.utils.error import ResponseErrorHandler


class Columns(BaseModel):
    prompt: str = "instruction"
    query: str = "input"
    response: str = "output"
    history: Union[str, None] = None
    messages: str = "conversations"
    system: Union[str, None] = None
    tools: Union[str, None] = None


class Tags(BaseModel):
    role_tag: str = "from"
    content_tag: str = "value"
    user_tag: str = "human"
    assistant_tag: str = "gpt"
    observation_tag: str = "observation"
    function_tag: str = "function_call"


class DatasetInfo(BaseModel):
    dataset_name: str
    load_from: Literal["file_name", "hf_hub_url"]
    dataset_src: str
    split: str = "train"
    num_samples: Union[int, None] = None
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    columns: Union[Columns, None] = Field(default_factory=Columns)
    tags: Union[Tags, None] = None

    @model_validator(mode="after")
    def check(self: "DatasetInfo") -> "DatasetInfo":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_]+", self.dataset_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'dataset_name' contain invalid characters",
                input={"dataset_name": self.dataset_name},
            )

        if self.load_from == "file_name":
            pattern = r"[^a-zA-Z0-9_\-\s\./]+"
        elif self.load_from == "hf_hub_url":
            pattern = r"[^a-zA-Z0-9_\-\./]+"

        if bool(re.search(pattern, self.dataset_src)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'dataset_src' contain invalid characters",
                input={"dataset_src": self.dataset_src},
            )

        if self.formatting == "alpaca" and self.tags is not None:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'tags' only used for 'sharegpt' formatting",
                input={"formatting": self.formatting},
            )
        elif self.formatting == "sharegpt" and self.tags is None:
            self.tags = Tags()

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class PostData(BaseModel):
    dataset_info: DatasetInfo
    dataset_file: Union[UploadFile, None]

    @model_validator(mode="after")
    def check(self: "PostData") -> "PostData":
        error_handler = ResponseErrorHandler()

        if self.dataset_file:
            if self.dataset_file.content_type != "application/json":
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'content_type' must be 'application/json'",
                    input={"dataset_file": f"{self.dataset_file.content_type}"},
                )

            if self.dataset_info.load_from != "file_name":
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_BODY],
                    msg="provide dataset_file, must load from 'file_name'",
                    input={"load_from": self.dataset_info.load_from},
                )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class GetData(BaseModel):
    dataset_name: str = ""

    @model_validator(mode="after")
    def check(self: "GetData") -> "GetData":
        error_handler = ResponseErrorHandler()

        if self.dataset_name:
            if bool(re.search(r"[^a-zA-Z0-9_]+", self.dataset_name)) is True:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_QUERY],
                    msg="'dataset_name' contain invalid characters",
                    input={"dataset_name": self.dataset_name},
                )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class PutData(BaseModel):
    dataset_name: str
    new_name: str

    @model_validator(mode="after")
    def check(self: "PutData") -> "PutData":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_]+", self.dataset_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'dataset_name' contain invalid characters",
                input={"dataset_name": self.dataset_name},
            )

        if bool(re.search(r"[^a-zA-Z0-9_]+", self.new_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'new_name' contain invalid characters",
                input={"new_name": self.new_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class DeleteData(BaseModel):
    dataset_name: str

    @model_validator(mode="after")
    def check(self: "DeleteData") -> "DeleteData":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_]+", self.dataset_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'dataset_name' contain invalid characters",
                input={"dataset_name": self.dataset_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self
