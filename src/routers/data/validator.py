from typing import Union

import orjson
from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.config.params import TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostData(BaseModel):
    dataset_name: str

    @model_validator(mode="after")
    def check(self: "PostData") -> "PostData":
        error_handler = ResponseErrorHandler()

        try:
            if redis_sync.client.hexists(TASK_CONFIG.data, self.dataset_name):
                raise ValueError("dataset_name already exists")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_FORM],
                msg=f"{e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetData(BaseModel):
    dataset_name: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetData") -> "GetData":
        error_handler = ResponseErrorHandler()

        try:
            if self.dataset_name and not redis_sync.client.hexists(
                TASK_CONFIG.data, self.dataset_name
            ):
                raise KeyError("dataset_name does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PutData(BaseModel):
    dataset_name: str
    new_name: str

    @model_validator(mode="after")
    def check(self: "PutData") -> "PutData":
        error_handler = ResponseErrorHandler()

        try:
            dataset_info = redis_sync.client.hget(TASK_CONFIG.data, self.dataset_name)

            if not dataset_info:
                raise KeyError("dataset_name does not exists")

            if orjson.loads(dataset_info)["is_used"] is True:
                raise ValueError("dataset is being used")

            if redis_sync.client.hexists(TASK_CONFIG.data, self.new_name):
                raise ValueError("new_name already in used")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"dataset_name": self.dataset_name, "new_name": self.new_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"dataset_name": self.dataset_name, "new_name": self.new_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class DelData(BaseModel):
    dataset_name: str

    @model_validator(mode="after")
    def check(self: "DelData") -> "DelData":
        error_handler = ResponseErrorHandler()

        try:
            dataset_info = redis_sync.client.hget(TASK_CONFIG.data, self.dataset_name)

            if not dataset_info:
                raise KeyError("dataset_name does not exists")

            if orjson.loads(dataset_info)["is_used"] is True:
                raise ValueError("dataset is being used")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"dataset_name": self.dataset_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
