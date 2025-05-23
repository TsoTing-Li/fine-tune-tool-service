from typing import Union

import orjson
from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, model_validator

from src.config.params import STATUS_CONFIG, TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostInferBackendStart(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str

    @model_validator(mode="after")
    def check(self: "PostInferBackendStart") -> "PostInferBackendStart":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.model_name)

            if not info:
                raise KeyError("model_name does not exists")

            info = orjson.loads(info)
            if info["container"]["infer_backend"]["status"] == STATUS_CONFIG.active:
                raise ValueError("model_name has been loaded")
            if info["last_model_path"] is None:
                raise KeyError("can not found model file")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"model_name": self.model_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"model_name": self.model_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"model_name": self.model_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostInferBackendStop(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str

    @model_validator(mode="after")
    def check(self: "PostInferBackendStop") -> "PostInferBackendStop":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.model_name)

            if not info:
                raise KeyError("model_name does not exists")

            if (
                orjson.loads(info)["container"]["infer_backend"]["status"]
                != STATUS_CONFIG.active
            ):
                raise KeyError("model_name is not being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"model_name": self.model_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"model_name": self.model_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetInferBackend(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetInferBackend") -> "GetInferBackend":
        error_handler = ResponseErrorHandler()

        try:
            if self.model_name is not None and not redis_sync.client.hexists(
                TASK_CONFIG.train, self.model_name
            ):
                raise KeyError("model_name does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={"model_name": self.model_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg="Database error",
                input={"model_name": self.model_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
