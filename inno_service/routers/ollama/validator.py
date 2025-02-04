import json
import os

import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStartOllama(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str

    @model_validator(mode="after")
    def check(self: "PostStartOllama") -> "PostStartOllama":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(self.model_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'model_name' does not exists",
                input={"model_name": self.model_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostStopOllama(BaseModel):
    ollama_container: str

    @model_validator(mode="after")
    def check(self: "PostStopOllama") -> "PostStopOllama":
        error_handler = ResponseErrorHandler()

        try:
            transport = httpx.HTTPTransport(uds="/var/run/docker.sock")
            with httpx.Client(transport=transport, timeout=None) as client:
                response = client.get(
                    "http://docker/containers/json",
                    params={"filters": json.dumps({"name": [self.ollama_container]})},
                )

            if response.status_code == 200:
                if response.json() == []:
                    error_handler.add(
                        type=error_handler.ERR_VALIDATE,
                        loc=[error_handler.LOC_BODY],
                        msg="'ollama_container' does not exists",
                        input={"ollama_container": self.ollama_container},
                    )
            else:
                error_handler.add(
                    type=error_handler.ERR_DOCKER,
                    loc=[error_handler.LOC_PROCESS],
                    msg=f"Error: {response.status_code}, {response.text}",
                    input={"ollama_container": self.ollama_container},
                )
        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_DOCKER,
                loc=[error_handler.LOC_PROCESS],
                msg=f"{e}",
                input={"ollama_container": self.ollama_container},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
