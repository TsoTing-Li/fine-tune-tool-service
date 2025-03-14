import re
from typing import Annotated, Union
from uuid import UUID

from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, StringConstraints, model_validator

from src.utils.error import ResponseErrorHandler


class PostDeploy(BaseModel):
    deploy_name: str
    device_uuid: UUID

    @model_validator(mode="after")
    def check(self: "PostDeploy") -> "PostDeploy":
        error_handler = ResponseErrorHandler()

        if self.device_uuid and self.device_uuid.version != 4:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="UUID version 4 expected",
                input={"device_uuid": str(self.device_uuid)},
            )

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", self.deploy_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'deploy_name' contain invalid characters",
                input={"deploy_name": self.deploy_name},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class GetHealthCheck(BaseModel):
    url: str

    @model_validator(mode="after")
    def check(self: "GetHealthCheck") -> "GetHealthCheck":
        error_handler = ResponseErrorHandler()

        valid_url = True
        ip_match = re.match(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", self.url)
        port_match = re.search(r":(\d+)", self.url)

        if ip_match:
            ip_parts = [int(part) for part in ip_match.groups()]
            if not all(0 <= part <= 255 for part in ip_parts):
                valid_url = False
        else:
            valid_url = False

        if port_match:
            port = int(port_match.group(1))
            if not (1 <= port <= 65535):
                valid_url = False
        else:
            valid_url = False

        if not valid_url:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="url contain invalid characters",
                input={"url": self.url},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostDevice(BaseModel):
    name: Annotated[str, StringConstraints(min_length=1)]
    url: Annotated[str, StringConstraints(min_length=9, max_length=21)]

    @model_validator(mode="after")
    def check(self: "PostDevice") -> "PostDevice":
        error_handler = ResponseErrorHandler()

        if bool(re.search(r"[^a-zA-Z0-9_\-/]+", self.name)) is True:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="name contain invalid characters",
                input={"name": self.name},
            )

        valid_url = True
        ip_match = re.match(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", self.url)
        port_match = re.search(r":(\d+)", self.url)

        if ip_match:
            ip_parts = [int(part) for part in ip_match.groups()]
            if not all(0 <= part <= 255 for part in ip_parts):
                valid_url = False
        else:
            valid_url = False

        if port_match:
            port = int(port_match.group(1))
            if not (1 <= port <= 65535):
                valid_url = False
        else:
            valid_url = False

        if not valid_url:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="url contain invalid characters",
                input={"url": self.url},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class GetDevice(BaseModel):
    uuid: Union[UUID, None]

    @model_validator(mode="after")
    def check(self: "GetDevice") -> "GetDevice":
        error_handler = ResponseErrorHandler()

        if self.uuid and self.uuid.version != 4:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="UUID version 4 expected",
                input={"uuid": str(self.uuid)},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PutDevice(BaseModel):
    uuid: UUID
    name: Annotated[Union[str, None], StringConstraints(min_length=1)] = None
    url: Annotated[Union[str, None], StringConstraints(min_length=9, max_length=21)] = (
        None
    )

    @model_validator(mode="after")
    def check(self: "PutDevice") -> "PutDevice":
        error_handler = ResponseErrorHandler()

        if self.uuid.version != 4:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="UUID version 4 expected",
                input={"uuid": str(self.uuid)},
            )

        if (
            self.name is not None
            and bool(re.search(r"[^a-zA-Z0-9_\-/]+", self.name)) is True
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="name contain invalid characters",
                input={"name": self.name},
            )

        if self.url:
            valid_url = True
            ip_match = re.match(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", self.url)
            port_match = re.search(r":(\d+)", self.url)

            if ip_match:
                ip_parts = [int(part) for part in ip_match.groups()]
                if not all(0 <= part <= 255 for part in ip_parts):
                    valid_url = False
            else:
                valid_url = False

            if port_match:
                port = int(port_match.group(1))
                if not (1 <= port <= 65535):
                    valid_url = False
            else:
                valid_url = False

            if not valid_url:
                error_handler.add(
                    type=error_handler.ERR_VALIDATE,
                    loc=[error_handler.LOC_QUERY],
                    msg="url contain invalid characters",
                    input={"url": self.url},
                )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class DelDevice(BaseModel):
    uuid: UUID

    @model_validator(mode="after")
    def check(self: "DelDevice") -> "DelDevice":
        error_handler = ResponseErrorHandler()

        if self.uuid.version != 4:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_QUERY],
                msg="UUID version 4 expected",
                input={"uuid": str(self.uuid)},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            ) from None

        return self
