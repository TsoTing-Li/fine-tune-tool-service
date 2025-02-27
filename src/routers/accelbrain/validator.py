from typing import Union

import orjson
from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, model_validator

from src import thirdparty
from src.config import params
from src.utils.error import ResponseErrorHandler


class PostDeploy(BaseModel):
    deploy_name: str
    accelbrain_url: str

    @model_validator(mode="after")
    def check(self: "PostDeploy") -> "PostDeploy":
        error_handler = ResponseErrorHandler()

        try:
            info = thirdparty.redis.handler.redis_sync.client.hget(
                params.TASK_CONFIG.train, self.deploy_name
            )

            if not info:
                raise ValueError("deploy_name does not exists")

            if not orjson.loads(info)["is_quantize"]:
                raise ValueError("deploy_name has not been quantified yet")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"deploy_name": self.deploy_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Unexpected error: {e}",
                input={
                    "deploy_name": self.deploy_name,
                    "accelbrain_url": self.accelbrain_url,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostDevice(BaseModel):
    accelbrain_device: str
    accelbrain_url: str

    @model_validator(mode="after")
    def check(self: "PostDevice") -> "PostDevice":
        error_handler = ResponseErrorHandler()

        try:
            if thirdparty.redis.handler.redis_sync.client.hexists(
                params.TASK_CONFIG.accelbrain_device, self.accelbrain_device
            ):
                raise ValueError("accelbrain_device already exists")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"accelbrain_device": self.accelbrain_device},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={
                    "accelbrain_device": self.accelbrain_device,
                    "accelbrain_url": self.accelbrain_url,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetDevice(BaseModel):
    accelbrain_device: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetDevice") -> "GetDevice":
        error_handler = ResponseErrorHandler()

        try:
            if (
                self.accelbrain_device
                and not thirdparty.redis.handler.redis_sync.client.hexists(
                    params.TASK_CONFIG.accelbrain_device, self.accelbrain_device
                )
            ):
                raise ValueError("accelbrain_device does not exists")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"accelbrain_device": self.accelbrain_device},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={
                    "accelbrain_device": self.accelbrain_device,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PutDevice(BaseModel):
    accelbrain_device: str
    accelbrain_url: str

    @model_validator(mode="after")
    def check(self: "PutDevice") -> "PutDevice":
        error_handler = ResponseErrorHandler()

        try:
            if not thirdparty.redis.handler.redis_sync.client.hexists(
                params.TASK_CONFIG.accelbrain_device, self.accelbrain_device
            ):
                raise ValueError("accelbrain_device does not exists")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"accelbrain_device": self.accelbrain_device},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={
                    "accelbrain_device": self.accelbrain_device,
                    "accelbrain_url": self.accelbrain_url,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class DelDevice(BaseModel):
    accelbrain_device: str

    @model_validator(mode="after")
    def check(self: "GetDevice") -> "GetDevice":
        error_handler = ResponseErrorHandler()

        try:
            if not thirdparty.redis.handler.redis_sync.client.hexists(
                params.TASK_CONFIG.accelbrain_device, self.accelbrain_device
            ):
                raise ValueError("accelbrain_device does not exists")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"accelbrain_device": self.accelbrain_device},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={
                    "accelbrain_device": self.accelbrain_device,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
