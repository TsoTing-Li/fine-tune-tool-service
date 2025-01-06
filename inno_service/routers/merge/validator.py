import json
import os

import httpx
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


class PostStartMerge(BaseModel):
    merge_name: str

    @model_validator(mode="after")
    def check(self: "PostStartMerge") -> "PostStartMerge":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(os.path.join(SAVE_PATH, self.merge_name)):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'merge_name' does not exists",
                input={"merge_name": self.merge_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PostStopMerge(BaseModel):
    merge_container: str

    @model_validator(mode="after")
    def check(self: "PostStopMerge") -> "PostStopMerge":
        error_handler = ResponseErrorHandler()

        try:
            transport = httpx.HTTPTransport(uds="/var/run/docker.sock")
            with httpx.Client(transport=transport, timeout=None) as client:
                response = client.get(
                    "http://docker/containers/json",
                    params={"filters": json.dumps({"name": [self.merge_container]})},
                )

            if response.status_code == 200:
                if response.json() == []:
                    error_handler.add(
                        type=error_handler.ERR_VALIDATE,
                        loc=[error_handler.LOC_BODY],
                        msg="'merge_container' does not exists",
                        input={"merge_container": self.merge_container},
                    )
            else:
                error_handler.add(
                    type=error_handler.ERR_DOCKER,
                    loc=[error_handler.LOC_PROCESS],
                    msg=f"Error: {response.status_code}, {response.text}",
                    input={"merge_container": self.merge_container},
                )
        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_DOCKER,
                loc=[error_handler.LOC_PROCESS],
                msg=f"{e}",
                input={"merge_container": self.merge_container},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
