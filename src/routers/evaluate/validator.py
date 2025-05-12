from typing import List

import orjson
from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.config.params import STATUS_CONFIG, TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostStartEval(BaseModel):
    eval_name: str

    @model_validator(mode="after")
    def check(self: "PostStartEval") -> "PostStartEval":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.eval_name)
            if not info:
                raise KeyError("eval_name does not exists")

            info = orjson.loads(info)

            if info["container"]["infer_backend"]["status"] != STATUS_CONFIG.active:
                raise ValueError("model has not been loaded")

            if info["container"]["eval"]["status"] == STATUS_CONFIG.active:
                raise ValueError("eval task is being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"eval_name": self.eval_name},
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
                input={"eval_name": self.eval_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"eval_name": self.eval_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostStopEval(BaseModel):
    eval_name: str

    @model_validator(mode="after")
    def check(self: "PostStopEval") -> "PostStopEval":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.eval_name)
            if not info:
                raise KeyError("eval_name dose not exists")

            if (
                orjson.loads(info)["container"]["eval"]["status"]
                != STATUS_CONFIG.active
            ):
                raise KeyError("eval task is not being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"eval_name": self.eval_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"eval_name": self.eval_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetEvalResult(BaseModel):
    eval_name: str

    @model_validator(mode="after")
    def check(self: "GetEvalResult") -> "GetEvalResult":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.eval_name)
            if info is None:
                raise KeyError("'eval_name' does not exists")

            eval_status = orjson.loads(info)["container"]["eval"]["status"]
            if eval_status != STATUS_CONFIG.finish:
                raise ValueError(f"can not get eval result, status is {eval_status}")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"eval_name": self.eval_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"eval_name": self.eval_name},
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
                input={"eval_name": self.eval_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class TaskInfo(BaseModel):
    name: str
    filter: str
    n_shot: int
    metric: str
    value: float
    stderr: float


class EvalResult(BaseModel):
    task_info: List[TaskInfo] = list()
