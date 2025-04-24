from typing import Union

from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.config.params import TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class GetSupportModel(BaseModel):
    base_model: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetSupportModel") -> "GetSupportModel":
        error_handler = ResponseErrorHandler()

        try:
            if self.base_model is not None and not redis_sync.client.hexists(
                TASK_CONFIG.support_model, self.base_model
            ):
                raise KeyError("base_model does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"base_model": self.base_model},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg="Database error",
                input={"base_model": self.base_model},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetEvalTask(BaseModel):
    eval_task: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetEvalTask") -> "GetEvalTask":
        error_handler = ResponseErrorHandler()

        try:
            if self.eval_task is not None and not redis_sync.client.hexists(
                TASK_CONFIG.eval_tasks, self.eval_task
            ):
                raise KeyError("eval_task does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"eval_task": self.eval_task},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg="Database error",
                input={"eval_task": self.eval_task},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
