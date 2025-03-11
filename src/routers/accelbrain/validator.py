from typing import Union

import orjson
from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, model_validator

from src.config.params import STATUS_CONFIG, TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostDeploy(BaseModel):
    deploy_name: str
    accelbrain_device: str

    @model_validator(mode="after")
    def check(self: "PostDeploy") -> "PostDeploy":
        error_handler = ResponseErrorHandler()

        try:
            train_info = redis_sync.client.hget(TASK_CONFIG.train, self.deploy_name)
            if not train_info:
                raise KeyError("deploy_name does not exists")

            accelbrain_info = redis_sync.client.hget(
                TASK_CONFIG.accelbrain_device, self.accelbrain_device
            )
            if not accelbrain_info:
                raise KeyError("accelbrain_device does not exists")

            if (
                orjson.loads(accelbrain_info)["deploy_status"][self.deploy_name]
                == STATUS_CONFIG.active
            ):
                raise ValueError("deploy_name is deploying to accelbrain_device ")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={
                    "deploy_name": self.deploy_name,
                    "accelbrain_device": self.accelbrain_device,
                },
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
                input={
                    "deploy_name": self.deploy_name,
                    "accelbrain_device": self.accelbrain_device,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Unexpected error: {e}",
                input={
                    "deploy_name": self.deploy_name,
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
            if redis_sync.client.hexists(
                TASK_CONFIG.accelbrain_device, self.accelbrain_device
            ):
                raise KeyError("accelbrain_device already exists")

        except KeyError as e:
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
            if self.accelbrain_device is not None and not redis_sync.client.hexists(
                TASK_CONFIG.accelbrain_device, self.accelbrain_device
            ):
                raise KeyError("accelbrain_device does not exists")

        except KeyError as e:
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

    @model_validator(mode="after")
    def check(self: "PutDevice") -> "PutDevice":
        error_handler = ResponseErrorHandler()

        try:
            accelbrain_info = redis_sync.client.hget(
                TASK_CONFIG.accelbrain_device, self.accelbrain_device
            )

            if not accelbrain_info:
                raise ValueError("accelbrain_device does not exists")

            if any(
                status == STATUS_CONFIG.active
                for status in orjson.loads(accelbrain_info)["deploy_status"].values()
            ):
                raise KeyError("accelbrain_device is executing deploy")

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

        except KeyError as e:
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
            accelbrain_info = redis_sync.client.hget(
                TASK_CONFIG.accelbrain_device, self.accelbrain_device
            )

            if not accelbrain_info:
                raise ValueError("accelbrain_device does not exists")

            if any(
                status == STATUS_CONFIG.active
                for status in orjson.loads(accelbrain_info)["deploy_status"].values()
            ):
                raise KeyError("accelbrain_device is executing deploy")

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

        except KeyError as e:
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
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
