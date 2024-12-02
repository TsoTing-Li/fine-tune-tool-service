import json

import httpx
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStartTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStartTrain") -> "PostStartTrain":
        error_handler = ResponseErrorHandler()

        try:
            transport = httpx.HTTPTransport(uds="/var/run/docker.sock")
            with httpx.Client(transport=transport, timeout=None) as client:
                response = client.get(
                    "http://docker/containers/json",
                    params={"filters": json.dumps({"name": [self.train_name]})},
                )

            if response.status_code == 200:
                if response.json() != []:
                    error_handler.add(
                        type=error_handler.ERR_VALIDATE,
                        loc=[error_handler.LOC_FORM],
                        msg="'train_name' already exists",
                        input={"train_name": self.train_name},
                    )
            else:
                error_handler.add(
                    type=error_handler.ERR_DOCKER,
                    loc=[error_handler.LOC_PROCESS],
                    msg=f"{response.status_code}\n{response.text}",
                    input={"train_name": self.train_name},
                )
        except BaseException as e:
            error_handler.add(
                type=error_handler.ERR_DOCKER,
                loc=[error_handler.LOC_PROCESS],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PostStopTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStartTrain") -> "PostStartTrain":
        error_handler = ResponseErrorHandler()

        try:
            transport = httpx.HTTPTransport(uds="/var/run/docker.sock")
            with httpx.Client(transport=transport, timeout=None) as client:
                response = client.get(
                    "http://docker/containers/json",
                    params={"filters": json.dumps({"name": [self.train_name]})},
                )

            if response.status_code == 200:
                if response.json() == []:
                    error_handler.add(
                        type=error_handler.ERR_VALIDATE,
                        loc=[error_handler.LOC_FORM],
                        msg="'train_name' does not exists",
                        input={"train_name": self.train_name},
                    )
            else:
                error_handler.add(
                    type=error_handler.ERR_DOCKER,
                    loc=[error_handler.LOC_PROCESS],
                    msg=f"{response.status_code}\n{response.text}",
                    input={"train_name": self.train_name},
                )
        except BaseException as e:
            error_handler.add(
                type=error_handler.ERR_DOCKER,
                loc=[error_handler.LOC_PROCESS],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
