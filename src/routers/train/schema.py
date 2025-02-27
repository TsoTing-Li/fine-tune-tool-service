import re
from typing import List, Literal, Union

from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.utils.error import ResponseErrorHandler


class DeepSpeedArgs(BaseModel):
    src: Literal["default", "file"]
    stage: Literal[2, 3, None] = None
    enable_offload: bool = False
    offload_device: Literal["cpu", "nvme", None] = None

    @model_validator(mode="after")
    def check(self: "DeepSpeedArgs") -> "DeepSpeedArgs":
        error_handler = ResponseErrorHandler()

        if self.src == "default" and not self.stage:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="must provide deepspeed stage '2' or '3', when 'src' is 'default'",
                input={"src": self.src, "stage": self.stage},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class Method(BaseModel):
    stage: Literal["sft"] = "sft"
    finetuning_type: Literal["full", "lora"]
    lora_target: Union[str, None] = None

    @model_validator(mode="after")
    def check(self: "Method") -> "Method":
        error_handler = ResponseErrorHandler()

        if self.finetuning_type == "lora" and not self.lora_target:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'lora_target' can not be empty when 'finetuning_type' is 'lora'",
                input={"lora_target": self.lora_target},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class Dataset(BaseModel):
    dataset: List[str]
    template: Literal["llama3", "gemma", "qwen", "mistral"]
    cutoff_len: int
    max_samples: int
    overwrite_cache: bool
    preprocessing_num_workers: int


class Output(BaseModel):
    logging_steps: int = 5
    save_steps: int = 5
    plot_loss: bool = False
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
                loc=[error_handler.LOC_BODY],
                msg="'per_device_train_batch_size' must be positive integer",
                input={"per_device_train_batch_size": self.per_device_train_batch_size},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class Val(BaseModel):
    val_size: float = 0.1
    per_device_eval_batch_size: int = 1
    eval_strategy: Literal["steps"] = "steps"

    @model_validator(mode="after")
    def check(self: "Val") -> "Val":
        error_handler = ResponseErrorHandler()

        if self.per_device_eval_batch_size <= 0:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'per_device_eval_batch_size' must be positive integer",
                input={"per_device_eval_batch_size": self.per_device_eval_batch_size},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

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
    val: Val = Field(default_factory=Val)

    @model_validator(mode="after")
    def check(self: "TrainArgs") -> "TrainArgs":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.model_name_or_path)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'model_name_or_path' contain invalid characters",
                input={"model_name_or_path": self.model_name_or_path},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostTrain(BaseModel):
    train_name: str
    train_args: TrainArgs
    deepspeed_args: Union[DeepSpeedArgs, None]
    deepspeed_file: Union[UploadFile, None]

    @model_validator(mode="after")
    def check(self: "PostTrain") -> "PostTrain":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.train_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'train_name' contain invalid characters",
                input={"train_name": self.train_name},
            )

        if self.deepspeed_args:
            if self.deepspeed_args.src == "file" and not self.deepspeed_file:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="must provide 'ds_file' when 'src' is 'file'",
                    input={
                        "deepspeed_args.src": self.deepspeed_args.src,
                        "deepspeed_file": self.deepspeed_file,
                    },
                )

            if (
                self.deepspeed_file
                and self.deepspeed_file.content_type != "application/json"
            ):
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'content_type' must be 'application/json'",
                    input={"deepspeed_file": self.deepspeed_file.content_type},
                )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class GetTrain(BaseModel):
    train_name: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetTrain") -> "GetTrain":
        error_handler = ResponseErrorHandler()

        if self.train_name:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.train_name):
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_QUERY],
                    msg="'train_name' contain invalid characters",
                    input={"train_name": self.train_name},
                )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PutTrain(BaseModel):
    train_name: str
    train_args: TrainArgs
    deepspeed_args: Union[DeepSpeedArgs, None]
    deepspeed_file: Union[UploadFile, None]

    @model_validator(mode="after")
    def check(self: "PutTrain") -> "PutTrain":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.train_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg="'train_name' contain invalid characters",
                input={"train_name": self.train_name},
            )

        if self.deepspeed_args:
            if self.deepspeed_args.src == "file" and not self.deepspeed_file:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="must provide 'ds_file' when 'src' is 'file'",
                    input={
                        "deepspeed_args.src": self.deepspeed_args.src,
                        "deepspeed_file": self.deepspeed_file,
                    },
                )

            if (
                self.deepspeed_file
                and self.deepspeed_file.content_type != "application/json"
            ):
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_FORM],
                    msg="'content_type' must be 'application/json'",
                    input={"deepspeed_file": self.deepspeed_file.content_type},
                )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class DelTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "DelTrain") -> "DelTrain":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.train_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="'train_name' contain invalid characters",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostStartTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStartTrain") -> "PostStartTrain":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.train_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'train_name' contain invalid characters",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostStopTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStopTrain") -> "PostStopTrain":
        error_handler = ResponseErrorHandler()

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.train_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'train_name' contain invalid characters",
                input={"train_name": self.train_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
