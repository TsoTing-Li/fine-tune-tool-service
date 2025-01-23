import os

from fastapi import status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostDeploy(BaseModel):
    deploy_name: str

    @model_validator(mode="after")
    def check(self: "PostDeploy") -> "PostDeploy":
        error_handler = ResponseErrorHandler()

        if not os.path.exists(self.deploy_name):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'deploy_name' does not exists",
                input={"deploy_name": os.path.basename(self.deploy_name)},
            )

        if not os.path.exists(f"{self.deploy_name}/quantize"):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'deploy_name' has not been quantified yet",
                input={"deploy_name": os.path.basename(self.deploy_name)},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
