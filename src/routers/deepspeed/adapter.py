from typing import Literal, Union

from pydantic import BaseModel


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
    nvme_path: Union[str, None] = None
    pin_memory: bool = True


class OffloadParam(BaseModel):
    device: Literal["cpu", "nvme"] = "cpu"
    nvme_path: Union[str, None] = None
    pin_memory: bool = True
    buffer_count: int = 20
    buffer_size: Union[int, float] = 1.5e8


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
    offload_optimizer: OffloadOptimizer
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
    offload_optimizer: OffloadOptimizer
    offload_param: OffloadParam
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
    zero_optimization: Z2_OFFLOAD_ZeroOptimization


class DS_Z2(DeepSpeedBaseModel):
    zero_optimization: Z2_ZeroOptimization


class DS_Z3_OFFLOAD(DeepSpeedBaseModel):
    zero_optimization: Z3_OFFLOAD_ZeroOptimization


class DS_Z3(DeepSpeedBaseModel):
    zero_optimization: Z3_ZeroOptimization


class PostDeepSpeedDefault(BaseModel):
    stage: int
    enable_offload: bool
    offload_device: Literal["cpu", "nvme", None] = None
    nvme_path: Union[str, None] = None

    def get_target_model(self):
        if self.stage == 2 and self.enable_offload:
            return DS_Z2_OFFLOAD(
                zero_optimization=Z2_OFFLOAD_ZeroOptimization(
                    offload_optimizer=OffloadOptimizer(
                        device=self.offload_device, nvme_path=self.nvme_path
                    )
                )
            )
        elif self.stage == 2 and not self.enable_offload:
            return DS_Z2(zero_optimization=Z2_ZeroOptimization())

        elif self.stage == 3 and self.enable_offload:
            return DS_Z3_OFFLOAD(
                zero_optimization=Z3_OFFLOAD_ZeroOptimization(
                    offload_optimizer=OffloadOptimizer(
                        device=self.offload_device, nvme_path=self.nvme_path
                    ),
                    offload_param=OffloadParam(
                        device=self.offload_device, nvme_path=self.nvme_path
                    ),
                )
            )

        elif self.stage == 3 and not self.enable_offload:
            return DS_Z3(zero_optimization=Z3_ZeroOptimization())
