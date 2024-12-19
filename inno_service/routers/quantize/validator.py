import json
import os

import httpx
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStartQuantize(BaseModel):
    checkpoint_path: str

    @model_validator(mode="after")
    def check(self: "PostStartQuantize") -> "PostStartQuantize":
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


class PostStopQuantize(BaseModel):
    quantize_container: str

    @model_validator(mode="after")
    def check(self: "PostStopQuantize") -> "PostStopQuantize":
        error_handler = ResponseErrorHandler()

        try:
            transport = httpx.HTTPTransport(uds="/var/run/docker.sock")
            with httpx.Client(transport=transport, timeout=None) as client:
                response = client.get(
                    "http://docker/containers/json",
                    params={"filters": json.dumps({"name": [self.quantize_container]})},
                )

            if response.status_code == 200:
                if response.json() == []:
                    error_handler.add(
                        type=error_handler.ERR_VALIDATE,
                        loc=[error_handler.LOC_BODY],
                        msg="'quantize_container' does not exists",
                        input={"quantize_container": self.quantize_container},
                    )
            else:
                error_handler.add(
                    type=error_handler.ERR_DOCKER,
                    loc=[error_handler.LOC_PROCESS],
                    msg=f"Error: {response.status_code}, {response.text}",
                    input={"quantize_container": self.quantize_container},
                )
        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_DOCKER,
                loc=[error_handler.LOC_PROCESS],
                msg=f"{e}",
                input={"quantize_container": self.quantize_container},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
