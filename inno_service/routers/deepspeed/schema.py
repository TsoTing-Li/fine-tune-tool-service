import re
from typing import Literal, Union

from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, model_validator

from inno_service.utils.error import ResponseErrorHandler


class OffloadOptimizer(BaseModel):
    device: Literal["cpu", "nvme"] = "cpu"
    nvme_path: str = None
    pin_memory: bool = True

    @model_validator(mode="after")
    def check(self: "OffloadOptimizer") -> "OffloadOptimizer":
        error_handler = ResponseErrorHandler()

        if self.device == "nvme" and not self.nvme_path:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'nvme_path' can not be empty",
                input={"nvme_path": self.nvme_path},
            )

        if self.nvme_path:
            if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.nvme_path)) is True:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'nvme_path' contain invalid characters",
                    input={"nvme_path": self.nvme_path},
                )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class OffloadParam(BaseModel):
    device: Literal["cpu", "nvme"] = "cpu"
    nvme_path: str = None
    pin_memory: bool = True
    buffer_count: int = 20
    buffer_size: Union[int, float] = 1.5e8

    @model_validator(mode="after")
    def check(self: "OffloadOptimizer") -> "OffloadOptimizer":
        error_handler = ResponseErrorHandler()

        if self.device == "nvme" and not self.nvme_path:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'nvme_path' can not be empty",
                input={"nvme_path": self.nvme_path},
            )

        if self.nvme_path:
            if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.nvme_path)) is True:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'nvme_path' contain invalid characters",
                    input={"nvme_path": self.nvme_path},
                )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class ZeroOptimization(BaseModel):
    stage: Literal[2, 3] = 3
    offload_optimizer: OffloadOptimizer = Field(default_factory=OffloadOptimizer)
    offload_param: OffloadParam = Field(default_factory=OffloadParam)
    allgather_partitions: bool = True
    allgather_bucket_size: Union[int, float] = 5e8
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    round_robin_gradients: bool = False
    sub_group_size: Union[int, float] = 1e9
    reduce_bucket_size: Union[str, int] = "auto"
    stage3_prefetch_bucket_size: Union[str, int] = "auto"
    stage3_param_persistence_threshold: Union[str, int] = "auto"
    stage3_max_live_parameters: Union[int, float] = 1e9
    stage3_max_reuse_distance: int = 0
    stage3_gather_16bit_weights_on_model_save: bool = True


class FP16(BaseModel):
    enabled: str = "auto"
    loss_scale: float = 0.0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: int = 1


class BF16(BaseModel):
    enabled: str = "auto"


class DeepSpeedBase(BaseModel):
    train_batch_size: str = "auto"
    train_micro_batch_size_per_gpu: str = "auto"
    gradient_accumulation_steps: str = "auto"
    gradient_clipping: str = "auto"
    zero_allow_untested_optimizer: bool = True
    fp16: FP16 = FP16()
    bf16: BF16 = BF16()


#     zero_optimization: ZeroOptimization = Field(default_factory=ZeroOptimization)


class DS_Z2_OFFLOAD(BaseModel):
    zero_optimization: ZeroOptimization = ZeroOptimization(stage=2)


class DS_Z2(BaseModel):
    zero_optimization: ZeroOptimization = ZeroOptimization(stage=2)


class DS_Z3_OFFLOAD(BaseModel):
    zero_optimization: ZeroOptimization = ZeroOptimization(stage=3)


class DS_Z3(BaseModel):
    zero_optimization: ZeroOptimization = ZeroOptimization(stage=3)


class PostDeepSpeedDefault(BaseModel):
    stage: Literal[2, 3]
    enable_offload: bool
    offload_device: Literal["cpu", "nvme", None] = None

    @model_validator(mode="after")
    def check(self: "PostDeepSpeedDefault") -> "PostDeepSpeedDefault":
        error_handler = ResponseErrorHandler()

        if self.enable_offload and not self.offload_device:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="must select offload device when enabled offload",
                input={
                    "enabled_offload": self.enable_offload,
                    "offload_device": self.offload_device,
                },
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PostDeepSpeedFile(BaseModel):
    ds_file: UploadFile

    @model_validator(mode="after")
    def check(self: "PostDeepSpeedFile") -> "PostDeepSpeedFile":
        error_handler = ResponseErrorHandler()

        if self.ds_file.content_type != "application/json":
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'content_type' must be 'application/json'",
                input={"ds_file": f"{self.ds_file.content_type}"},
            )

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.ds_file.filename)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ds_file filename' contain invalid characters",
                input={"ds_file filename": self.ds_file.filename},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class GetDeepSpeedPreview(BaseModel):
    ds_file_name: str

    @model_validator(mode="after")
    def check(self: "GetDeepSpeedPreview") -> "GetDeepSpeedPreview":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.ds_file_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ds_file_name' contain invalid characters",
                input={"dataset_src": self.ds_file_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class DelDeepSpeed(BaseModel):
    ds_file_name: str

    @model_validator(mode="after")
    def check(self: "DelDeepSpeed") -> "DelDeepSpeed":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.ds_file_name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'ds_file_name' contain invalid characters",
                input={"dataset_src": self.ds_file_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
