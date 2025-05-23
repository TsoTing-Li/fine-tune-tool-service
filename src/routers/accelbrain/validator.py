from typing import Annotated, Dict, Union
from uuid import UUID

import orjson
from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, model_validator
from pydantic.types import UuidVersion

from src.config.params import STATUS_CONFIG, TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostDeploy(BaseModel):
    deploy_name: str
    device_uuid: UUID

    @model_validator(mode="after")
    def check(self: "PostDeploy") -> "PostDeploy":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists(TASK_CONFIG.train, self.deploy_name):
                raise KeyError("deploy_name does not exists")

            accelbrain_info = redis_sync.client.hget(
                TASK_CONFIG.accelbrain_device, str(self.device_uuid)
            )
            if not accelbrain_info:
                raise KeyError("device_uuid does not exists")

            deploy_status = redis_sync.client.hget(
                TASK_CONFIG.deploy, f"{self.deploy_name}-{self.device_uuid}"
            )
            if deploy_status:
                if orjson.loads(deploy_status)["status"] == STATUS_CONFIG.active:
                    raise ValueError("deploy_name is deploying to accelbrain_device")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={
                    "deploy_name": self.deploy_name,
                    "device_uuid": str(self.device_uuid),
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
                    "device_uuid": str(self.device_uuid),
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
                    "device_uuid": str(self.device_uuid),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetDeploy(BaseModel):
    deploy_name: Union[str, None]
    device_uuid: Union[UUID, None]

    @model_validator(mode="after")
    def check(self: "GetDeploy") -> "GetDeploy":
        error_handler = ResponseErrorHandler()

        if (self.deploy_name is None) != (self.device_uuid is None):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="Both deploy_name and device_uuid must be either None or non-None",
                input={
                    "deploy_name": self.deploy_name,
                    "device_uuid": str(self.device_uuid),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_handler.errors
            ) from None

        try:
            if self.deploy_name is not None and not redis_sync.client.hexists(
                TASK_CONFIG.train, self.deploy_name
            ):
                raise KeyError("deploy_name does not exists")

            if self.device_uuid is not None and not redis_sync.client.hexists(
                TASK_CONFIG.accelbrain_device, str(self.device_uuid)
            ):
                raise KeyError("device_uuid does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg=f"{e}",
                input={
                    "deploy_name": self.deploy_name,
                    "device_uuid": str(self.device_uuid),
                },
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
                    "deploy_name": self.deploy_name,
                    "device_uuid": str(self.device_uuid),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostDevice(BaseModel):
    name: str
    url: str

    @model_validator(mode="after")
    def check(self: "PostDevice") -> "PostDevice":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hgetall(TASK_CONFIG.accelbrain_device)
            for value in info.values():
                value = orjson.loads(value)
                if value["name"] == self.name:
                    raise KeyError("name already exists")

                if value["url"] == self.url:
                    raise ValueError("url already exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"name": self.name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"url": self.url},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={"name": self.name, "url": self.url},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetDevice(BaseModel):
    uuid: Union[Annotated[UUID, UuidVersion(4)], None]

    @model_validator(mode="after")
    def check(self: "GetDevice") -> "GetDevice":
        error_handler = ResponseErrorHandler()

        try:
            if self.uuid is not None and not redis_sync.client.hexists(
                TASK_CONFIG.accelbrain_device, str(self.uuid)
            ):
                raise KeyError("uuid does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"uuid": str(self.uuid)},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={"uuid": str(self.uuid) if self.uuid is not None else None},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PutDevice(BaseModel):
    uuid: Annotated[UUID, UuidVersion(4)]
    name: Union[str, None]
    url: Union[str, None]

    @model_validator(mode="after")
    def check(self: "PutDevice") -> "PutDevice":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hgetall(TASK_CONFIG.accelbrain_device)

            accelbrain_info = info.get(str(self.uuid), None)
            if not accelbrain_info:
                raise ValueError("uuid does not exists")

            for value in info.values():
                value = orjson.loads(value)
                if self.name and value["name"] == self.name:
                    raise KeyError("name already exists")

                if self.url and value["url"] == self.url:
                    raise KeyError("url already exists")

            deploy_status: Dict[str, str] = redis_sync.client.hgetall(
                TASK_CONFIG.deploy
            )
            if any(
                orjson.loads(v)["status"] == STATUS_CONFIG.active
                for k, v in deploy_status.items()
                if k.endswith(str(self.uuid))
            ):
                raise KeyError("accelbrain_device is executing deploy")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"uuid": str(self.uuid)},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"name": self.name, "url": self.url},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={"uuid": str(self.uuid), "name": self.name, "url": self.url},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class DelDevice(BaseModel):
    uuid: Annotated[UUID, UuidVersion(4)]

    @model_validator(mode="after")
    def check(self: "DelDevice") -> "DelDevice":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists(
                TASK_CONFIG.accelbrain_device, str(self.uuid)
            ):
                raise ValueError("uuid does not exists")

            deploy_status: Dict[str, str] = redis_sync.client.hgetall(
                TASK_CONFIG.deploy
            )
            if any(
                orjson.loads(v)["status"] == STATUS_CONFIG.active
                for k, v in deploy_status.items()
                if k.endswith(str(self.uuid))
            ):
                raise KeyError("accelbrain_device is executing deploy")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"uuid": str(self.uuid)},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"uuid": str(self.uuid)},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"{e}",
                input={"uuid": str(self.uuid)},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
