import json
import os

import httpx
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


class PostTrain(BaseModel):
    train_path: str

    @model_validator(mode="after")
    def check(self: "PostTrain") -> "PostTrain":
        error_handler = ResponseErrorHandler()

        if os.path.exists(self.train_path):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'train_path' already exists",
                input={"train_path": self.train_path},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class GetTrain(BaseModel):
    train_path: str

    @model_validator(mode="after")
    def check(self: "GetTrain") -> "GetTrain":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(self.train_path):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'train_path' does not exists",
                input={"train_path": self.train_path},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PutTrain(BaseModel):
    train_path: str

    @model_validator(mode="after")
    def check(self: "PutTrain") -> "PutTrain":
        error_handler = ResponseErrorHandler()
        if not os.path.exists(self.train_path):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'train_path' does not exists",
                input={"train_path": self.train_path},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class DelTrain(BaseModel):
    train_path: str

    @model_validator(mode="after")
    def check(self: "DelTrain") -> "DelTrain":
        error_handler = ResponseErrorHandler()
        if not os.path.exists(self.train_path):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'train_path' does not exists",
                input={"train_path": self.train_path},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class PostStartTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStartTrain") -> "PostStartTrain":
        error_handler = ResponseErrorHandler()

        if os.path.exists(os.path.join(SAVE_PATH, self.train_name)):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'train_name' already exists",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PostStopTrain(BaseModel):
    train_container: str

    @model_validator(mode="after")
    def check(self: "PostStopTrain") -> "PostStopTrain":
        error_handler = ResponseErrorHandler()

        try:
            transport = httpx.HTTPTransport(uds="/var/run/docker.sock")
            with httpx.Client(transport=transport, timeout=None) as client:
                response = client.get(
                    "http://docker/containers/json",
                    params={"filters": json.dumps({"name": [self.train_container]})},
                )

            if response.status_code == 200:
                if response.json() == []:
                    error_handler.add(
                        type=error_handler.ERR_VALIDATE,
                        loc=[error_handler.LOC_BODY],
                        msg="'train_container' does not exists",
                        input={"train_container": self.train_container},
                    )
            else:
                error_handler.add(
                    type=error_handler.ERR_DOCKER,
                    loc=[error_handler.LOC_PROCESS],
                    msg=f"Error: {response.status_code}, {response.text}",
                    input={"train_container": self.train_container},
                )
        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_DOCKER,
                loc=[error_handler.LOC_PROCESS],
                msg=f"{e}",
                input={"train_container": self.train_container},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
