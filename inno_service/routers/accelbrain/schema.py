import re

from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostDeploy(BaseModel):
    deploy_name: str

    @model_validator(mode="after")
    def check(self: "PostDeploy") -> "PostDeploy":
        error_handler = ResponseErrorHandler()

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


class GetHealthcheck(BaseModel):
    accelbrain_url: str

    @model_validator(mode="after")
    def check(self: "GetHealthcheck") -> "GetHealthcheck":
        error_handler = ResponseErrorHandler()

        valid_url = True
        ip_match = re.match(
            r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", self.accelbrain_url
        )
        port_match = re.search(r":(\d+)", self.accelbrain_url)

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
                msg="'accelbrain_url' contain invalid characters",
                input={"accelbrain_url": self.accelbrain_url},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class PostSaveurl(BaseModel):
    accelbrain_url: str

    @model_validator(mode="after")
    def check(self: "PostSaveurl") -> "PostSaveurl":
        error_handler = ResponseErrorHandler()

        valid_url = True
        ip_match = re.match(
            r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", self.accelbrain_url
        )
        port_match = re.search(r":(\d+)", self.accelbrain_url)

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
                msg="'accelbrain_url' contain invalid characters",
                input={"accelbrain_url": self.accelbrain_url},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
