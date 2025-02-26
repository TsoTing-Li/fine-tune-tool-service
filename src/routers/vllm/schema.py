import re

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, model_validator

from src.utils.error import ResponseErrorHandler


class PostStartVLLM(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    image_name: str
    service_port: int
    docker_network_name: str
    model_name: str
    local_safetensors_path: str
    base_model: str
    hf_home: str
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 8192
    cpu_offload_gb: int = 0
    tensor_parallel_size: int = 1

    @model_validator(mode="after")
    def check(self: "PostStartVLLM") -> "PostStartVLLM":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.model_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'model_name' contain invalid characters",
                input={"model_name": self.model_name},
            )

        if 0 > self.gpu_memory_utilization or 1 < self.gpu_memory_utilization:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'gpu_memory_utilization' must between 0 to 1",
                input={"gpu_memory_utilization": self.gpu_memory_utilization},
            )

        if self.tensor_parallel_size < 0:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'tensor_parallel_size' must larger than 0",
                input={"tensor_parallel_size": self.tensor_parallel_size},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostStopVLLM(BaseModel):
    vllm_container: str

    @model_validator(mode="after")
    def check(self: "PostStopVLLM") -> "PostStopVLLM":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.vllm_container):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'vllm_container' contain invalid characters",
                input={"vllm_container": self.vllm_container},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
