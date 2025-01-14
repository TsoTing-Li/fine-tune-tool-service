import re
from typing import Literal, Union

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class FP16(BaseModel):
    enabled: str = "auto"
    loss_scale: float = 0.0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: int = 1


class BF16(BaseModel):
    enabled: str = "auto"


class DeepSpeedBaseModel(BaseModel):
    train_batch_size: str = "auto"
    train_micro_batch_size_per_gpu: str = "auto"
    gradient_accumulation_steps: str = "auto"
    gradient_clipping: str = "auto"
    zero_allow_untested_optimizer: bool = True
    fp16: FP16 = FP16()
    bf16: BF16 = BF16()


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


class Z2_ZeroOptimization(BaseModel):
    stage: int = 2
    allgather_partitions: bool = True
    allgather_bucket_size: Union[int, float] = 5e8
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    round_robin_gradients: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: Union[int, float] = 5e8


class Z2_OFFLOAD_ZeroOptimization(BaseModel):
    stage: int = 2
    offload_optimizer: OffloadOptimizer = OffloadOptimizer()
    allgather_partitions: bool = True
    allgather_bucket_size: Union[int, float] = 5e8
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    round_robin_gradients: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: Union[int, float] = 5e8


class Z3_ZeroOptimization(BaseModel):
    stage: int = 3
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    sub_group_size: Union[int, float] = 1e9
    reduce_bucket_size: Union[str, int] = "auto"
    stage3_prefetch_bucket_size: Union[str, int] = "auto"
    stage3_param_persistence_threshold: Union[str, int] = "auto"
    stage3_max_live_parameters: Union[int, float] = 1e9
    stage3_max_reuse_distance: int = 0
    stage3_gather_16bit_weights_on_model_save: bool = True


class Z3_OFFLOAD_ZeroOptimization(BaseModel):
    stage: int = 3
    offload_optimizer: OffloadOptimizer = OffloadOptimizer()
    offload_param: OffloadParam = OffloadParam()
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    sub_group_size: Union[int, float] = 1e9
    reduce_bucket_size: Union[str, int] = "auto"
    stage3_prefetch_bucket_size: Union[str, int] = "auto"
    stage3_param_persistence_threshold: Union[str, int] = "auto"
    stage3_max_live_parameters: Union[int, float] = 1e9
    stage3_max_reuse_distance: int = 0
    stage3_gather_16bit_weights_on_model_save: bool = True


class DS_Z2_OFFLOAD(DeepSpeedBaseModel):
    zero_optimization: Z2_OFFLOAD_ZeroOptimization = Z2_OFFLOAD_ZeroOptimization()


class DS_Z2(DeepSpeedBaseModel):
    zero_optimization: Z2_ZeroOptimization = Z2_ZeroOptimization()


class DS_Z3_OFFLOAD(DeepSpeedBaseModel):
    zero_optimization: Z3_OFFLOAD_ZeroOptimization = Z3_OFFLOAD_ZeroOptimization()


class DS_Z3(DeepSpeedBaseModel):
    zero_optimization: Z3_ZeroOptimization = Z3_ZeroOptimization()


class PostDeepSpeedDefault(BaseModel):
    stage: int
    enable_offload: bool

    def get_target_model(self):
        if self.stage == 2 and self.enable_offload:
            return DS_Z2_OFFLOAD()
        elif self.stage == 2 and not self.enable_offload:
            return DS_Z2()
        elif self.stage == 3 and self.enable_offload:
            return DS_Z3_OFFLOAD()
        elif self.stage == 3 and not self.enable_offload:
            return DS_Z3()
