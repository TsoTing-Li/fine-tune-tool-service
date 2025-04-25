import os
from typing import List, Union

import orjson
from fastapi import HTTPException, status
from pydantic import BaseModel, Field, model_validator

from src.config.params import COMMON_CONFIG, STATUS_CONFIG, TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostTrain") -> "PostTrain":
        error_handler = ResponseErrorHandler()

        try:
            if redis_sync.client.hexists(TASK_CONFIG.train, self.train_name):
                raise ValueError("train_name already exists")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetTrain(BaseModel):
    train_name: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetTrain") -> "GetTrain":
        error_handler = ResponseErrorHandler()

        try:
            if self.train_name and not redis_sync.client.hexists(
                TASK_CONFIG.train, self.train_name
            ):
                raise KeyError("train_name does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PutTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PutTrain") -> "PutTrain":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.train_name)
            if not info:
                raise KeyError("train_name does not exists")

            info = orjson.loads(info)
            if info["container"]["train"]["status"] == STATUS_CONFIG.active:
                raise ValueError("train_name is being executed")
            if info["container"]["infer_backend"]["status"] == STATUS_CONFIG.active:
                raise ValueError("train_name is being inferred")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class DelTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "DelTrain") -> "DelTrain":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.train_name)
            if not info:
                raise KeyError("train_name does not exists")

            info = orjson.loads(info)
            if info["container"]["train"]["status"] == STATUS_CONFIG.active:
                raise ValueError("train_name is being executed")
            if info["container"]["infer_backend"]["status"] == STATUS_CONFIG.active:
                raise ValueError("train_name is being inferred")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostStartTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStartTrain") -> "PostStartTrain":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists(
                TASK_CONFIG.train, self.train_name
            ) or not os.path.exists(
                os.path.join(COMMON_CONFIG.save_path, self.train_name)
            ):
                raise KeyError("train_name does not exists")

            info = redis_sync.client.hget(TASK_CONFIG.train, self.train_name)
            if (
                orjson.loads(info)["container"]["train"]["status"]
                == STATUS_CONFIG.active
            ):
                raise ValueError("train_name is being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostStopTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStopTrain") -> "PostStopTrain":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.train_name)
            if not info:
                raise KeyError("train_name does not exists")

            if (
                orjson.loads(info)["container"]["train"]["status"]
                != STATUS_CONFIG.active
            ):
                raise KeyError("train_name is not being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None


class GetTrainLog(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "GetTrainLog") -> "GetTrainLog":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.train_name)
            if info is None:
                raise KeyError("train_name does not exists")

            train_status = orjson.loads(info)["container"]["train"]["status"]
            if train_status in {STATUS_CONFIG.setup, STATUS_CONFIG.active}:
                raise ValueError(f"can not get train log, status is {train_status}")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetTrainResult(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "GetTrainResult") -> "GetTrainResult":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.train_name)
            if info is None:
                raise KeyError("train_name does not exists")

            train_status = orjson.loads(info)["container"]["train"]["status"]
            if train_status in {STATUS_CONFIG.setup, STATUS_CONFIG.active}:
                raise ValueError(f"can not get train result, status is {train_status}")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class LogHistory(BaseModel):
    epoch: float
    step: int
    loss: float
    eval_loss: float


class FinalReport(BaseModel):
    epoch: Union[float, None] = None
    step: Union[int, None] = None
    total_flos: Union[float, None] = None
    train_loss: Union[float, None] = None
    train_runtime: Union[float, None] = None
    train_samples_per_second: Union[float, None] = None
    train_steps_per_second: Union[float, None] = None
    eval_loss: Union[float, None] = None
    eval_runtime: Union[float, None] = None
    eval_samples_per_second: Union[float, None] = None
    eval_steps_per_second: Union[float, None] = None


class TrainResult(BaseModel):
    log_history: List[LogHistory] = list()
    final_report: FinalReport = Field(default_factory=FinalReport)
