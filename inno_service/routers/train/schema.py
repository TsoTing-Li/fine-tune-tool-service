import re
from typing import Union

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal

from inno_service.utils.error import ResponseErrorHandler


class Method(BaseModel):
    stage: Literal["sft"] = "sft"
    do_train: bool = True
    finetuning_type: Literal["full", "lora"]
    lora_target: Union[str, None] = None
    deepspeed: Union[str, None] = None

    @model_validator(mode="after")
    def check(self: "Method") -> "Method":
        error_handler = ResponseErrorHandler()

        if (
            self.deepspeed
            and bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.deepspeed)) is True
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'deepspeed' contain invalid characters",
                input={"deepspeed": self.deepspeed},
            )

        if self.finetuning_type == "lora" and not self.lora_target:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'lora_target' can not be empty when 'finetuning_type' is 'lora'",
                input={"lora_target": self.lora_target},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class Dataset(BaseModel):
    dataset: str
    template: Literal["llama3", "gemma", "qwen", "mistral"]
    cutoff_len: int
    max_samples: int
    overwrite_cache: bool
    preprocessing_num_workers: int


class Output(BaseModel):
    logging_steps: int
    save_steps: int
    plot_loss: bool
    overwrite_output_dir: bool = False
    log_level: str = "info"
    logging_first_step: bool = True


class Params(BaseModel):
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    lr_scheduler_type: Literal[
        "cosine",
        "cosine_with_restarts",
        "cosine_with_min_lr",
        "constant",
        "constant_with_warmup",
        "inverse_sqrt",
        "linear",
        "polynomial",
        "reduce_lr_on_plateau",
        "warmup_stable_decay",
    ]
    warmup_ratio: float
    bf16: bool = True
    ddp_timeout: int

    @model_validator(mode="after")
    def check(self: "Params") -> "Params":
        error_handler = ResponseErrorHandler()

        if self.per_device_train_batch_size <= 0:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'per_device_train_batch_size' must be positive integer",
                input={"per_device_train_batch_size": self.per_device_train_batch_size},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class Eval(BaseModel):
    val_size: float = 0.1
    per_device_eval_batch_size: int = 1
    eval_strategy: Literal["steps"] = "steps"

    @model_validator(mode="after")
    def check(self: "Eval") -> "Eval":
        error_handler = ResponseErrorHandler()

        if self.per_device_eval_batch_size <= 0:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'per_device_eval_batch_size' must be positive integer",
                input={"per_device_eval_batch_size": self.per_device_eval_batch_size},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self


class TrainArgs(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name_or_path: str
    method: Method = Field(default_factory=Method)
    dataset: Dataset = Field(default_factory=Dataset)
    output: Output = Field(default_factory=Output)
    params: Params = Field(default_factory=Params)
    eval: Eval = Field(default_factory=Eval)

    @model_validator(mode="after")
    def check(self: "TrainArgs") -> "TrainArgs":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.model_name_or_path)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'model_name_or_path' contain invalid characters",
                input={"model_name_or_path": self.model_name_or_path},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PostStartTrain(BaseModel):
    train_name: Union[str, None] = None
    train_args: TrainArgs = Field(default_factory=TrainArgs)

    @model_validator(mode="after")
    def check(self: "PostStartTrain") -> "PostStartTrain":
        error_handler = ResponseErrorHandler()

        if (
            self.train_name
            and bool(re.search(r"[^a-zA-Z0-9_\-\s]+", self.train_name)) is True
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'train_name' contain invalid characters",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self


class PostStopTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStopTrain") -> "PostStopTrain":
        error_handler = ResponseErrorHandler()

        if (
            self.train_name
            and bool(re.search(r"[^a-zA-Z0-9_\-\s]+", self.train_name)) is True
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'train_name' contain invalid characters",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)
        return self
