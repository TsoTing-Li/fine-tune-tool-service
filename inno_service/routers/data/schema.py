import re
from typing import Literal, Union

from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class Columns(BaseModel):
    prompt: Union[str, None] = None
    query: Union[str, None] = None
    response: Union[str, None] = None
    history: Union[str, None] = None
    messages: Union[str, None] = None
    system: Union[str, None] = None
    tools: Union[str, None] = None


class Tags(BaseModel):
    role_tag: str
    content_tag: str
    user_tag: str
    assistant_tag: str
    observation_tag: Union[str, None] = None
    function_tag: Union[str, None] = None
    system_tag: Union[str, None] = None

    @model_validator(mode="after")
    def check(self: "Tags") -> "Tags":
        error_handler = ResponseErrorHandler()

        if (self.observation_tag and not self.function_tag) or (
            not self.observation_tag and self.function_tag
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="observation_tag and function_tag must exist at the same time",
                input={
                    "observation_tag": self.observation_tag,
                    "function_tag": self.function_tag,
                },
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class DatasetInfo(BaseModel):
    dataset_name: str
    load_from: Literal["file_name", "hf_hub_url"]
    dataset_src: str
    subset: Union[str, None] = None
    split: str = "train"
    num_samples: Union[int, None] = None
    formatting: Literal["alpaca", "sharegpt"]
    columns: Columns
    tags: Union[Tags, None] = None

    @model_validator(mode="after")
    def check(self: "DatasetInfo") -> "DatasetInfo":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-/]+", self.dataset_name)) is True:
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
                loc=[error_handler.LOC_FORM],
                msg="'dataset_src' contain invalid characters",
                input={"dataset_src": self.dataset_src},
            )

        if self.subset:
            if bool(re.search(pattern, self.subset)) is True:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'subset' contain invalid characters",
                    input={"subset": self.subset},
                )

            if self.load_from != "hf_hub_url":
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'subset' only use in 'hf_hub_url'",
                    input={"load_from": self.load_from, "subset": self.subset},
                )

        if bool(re.search(r"[^a-zA-Z0-9_\-]+", self.split)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'split contain invalid characters",
                input={"split": self.split},
            )

        if self.formatting == "alpaca" and self.tags is not None:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'tags' only used for 'sharegpt' formatting",
                input={"formatting": self.formatting},
            )

        if self.formatting == "sharegpt" and self.tags is None:
            self.tags = Tags()

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostData(BaseModel):
    dataset_info: DatasetInfo
    dataset_file: Union[UploadFile, None]

    @model_validator(mode="after")
    def check(self: "PostData") -> "PostData":
        error_handler = ResponseErrorHandler()

        if self.dataset_info.load_from == "file_name":
            if not self.dataset_file:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="provide dataset_file, must load from 'file_name'",
                    input={"load_from": self.dataset_info.load_from},
                )
            elif self.dataset_file.content_type != "application/json":
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'content_type' must be 'application/json'",
                    input={"dataset_file": f"{self.dataset_file.content_type}"},
                )
        else:
            if self.dataset_file:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="provide 'dataset_file' only if 'load_from' is 'file_name'",
                    input={
                        "dataset_file": f"{self.dataset_file.filename}",
                        "dataset_info.load_from": self.dataset_info.load_from,
                    },
                )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class GetData(BaseModel):
    dataset_name: str = ""

    @model_validator(mode="after")
    def check(self: "GetData") -> "GetData":
        error_handler = ResponseErrorHandler()

        if self.dataset_name:
            if bool(re.search(r"[^a-zA-Z0-9_\-/]+", self.dataset_name)) is True:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_QUERY],
                    msg="'dataset_name' contain invalid characters",
                    input={"dataset_name": self.dataset_name},
                )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PutData(BaseModel):
    dataset_name: str
    new_name: str

    @model_validator(mode="after")
    def check(self: "PutData") -> "PutData":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-/]+", self.dataset_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'dataset_name' contain invalid characters",
                input={"dataset_name": self.dataset_name},
            )

        if bool(re.search(r"[^a-zA-Z0-9_\-/]+", self.new_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'new_name' contain invalid characters",
                input={"new_name": self.new_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class DeleteData(BaseModel):
    dataset_name: str

    @model_validator(mode="after")
    def check(self: "DeleteData") -> "DeleteData":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-/]+", self.dataset_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'dataset_name' contain invalid characters",
                input={"dataset_name": self.dataset_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
