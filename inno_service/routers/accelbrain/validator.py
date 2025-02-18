import orjson
from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, model_validator

from inno_service import thirdparty
from inno_service.utils.error import ResponseErrorHandler


class PostDeploy(BaseModel):
    deploy_name: str
    accelbrain_url: str

    @model_validator(mode="after")
    def check(self: "PostDeploy") -> "PostDeploy":
        error_handler = ResponseErrorHandler()

        try:
            info = thirdparty.redis.handler.redis_sync.client.hget(
                "TRAIN", self.deploy_name
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
