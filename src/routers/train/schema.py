import re
from typing import List, Literal, Union

from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel, model_validator

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
    stage: Literal["sft"]
    finetuning_type: Literal["full", "lora"]


class Dataset(BaseModel):
    dataset: List[str]
    template: str
    cutoff_len: int
    max_samples: int
    overwrite_cache: bool
    preprocessing_num_workers: int


class Output(BaseModel):
    logging_steps: int
    save_steps: int
    plot_loss: bool
    overwrite_output_dir: bool
    log_level: str
    logging_first_step: bool


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
    compute_type: Literal["bf16", "fp16"]
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
    val_size: float
    per_device_eval_batch_size: int
    eval_strategy: Literal["steps"]

    @model_validator(mode="after")
    def check(self: "Val") -> "Val":
        error_handler = ResponseErrorHandler()

        if self.val_size > 1.0 or self.val_size < 0.1:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="val_size must be between 0.1 and 1.0",
                input={"val_size": self.val_size},
            )

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


class Lora(BaseModel):
    lora_alpha: Union[int, None]
    lora_dropout: Union[float, None]
    lora_rank: Union[int, None]
    lora_target: Union[List[str], None]


class TrainArgs(BaseModel):
    base_model: str
    method: Method
    dataset: Dataset
    output: Output
    params: Params
    val: Val
    lora: Union[Lora, None]

    @model_validator(mode="after")
    def check(self: "TrainArgs") -> "TrainArgs":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.base_model)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'base_model' contain invalid characters",
                input={"base_model": self.base_model},
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

        if (
            self.train_args.method.finetuning_type == "lora"
            and not self.train_args.lora
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="lora params can not be empty when 'finetuning_type' is 'lora'",
                input={"lora": self.train_args.lora},
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

        if (
            self.train_args.method.finetuning_type == "lora"
            and not self.train_args.lora
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="lora params can not be empty when 'finetuning_type' is 'lora'",
                input={"lora": self.train_args.lora},
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
