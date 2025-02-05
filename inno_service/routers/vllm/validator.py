import json
import os

import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStartVLLM(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str

    @model_validator(mode="after")
    def check(self: "PostStartVLLM") -> "PostStartVLLM":
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
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            )

        return self


class PostStopVLLM(BaseModel):
    vllm_container: str

    @model_validator(mode="after")
    def check(self: "PostStopVLLM") -> "PostStopVLLM":
        error_handler = ResponseErrorHandler()

        try:
            transport = httpx.HTTPTransport(uds="/var/run/docker.sock")
            with httpx.Client(transport=transport, timeout=None) as client:
                response = client.get(
                    "http://docker/containers/json",
                    params={"filters": json.dumps({"name": [self.vllm_container]})},
                )

            if response.status_code == 200:
                if response.json() == []:
                    error_handler.add(
                        type=error_handler.ERR_VALIDATE,
                        loc=[error_handler.LOC_BODY],
                        msg="'vllm_container' does not exists",
                        input={"vllm_container": self.vllm_container},
                    )
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=error_handler.errors,
                    ) from None
            else:
                raise RuntimeError(f"{response.json()['message']}")

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_DOCKER,
                loc=[error_handler.LOC_PROCESS],
                msg=f"Unexpected error: {e}",
                input={"vllm_container": self.vllm_container},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
