from typing import Union
from uuid import UUID

from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.utils.error import ResponseErrorHandler


class GetSupportModel(BaseModel):
    support_model_uuid: Union[UUID, None]

    @model_validator(mode="after")
    def check(self: "GetSupportModel") -> "GetSupportModel":
        error_handler = ResponseErrorHandler()

        if self.support_model_uuid is not None and self.support_model_uuid.version != 4:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="UUID version 4 expected",
                input={"support_model_uuid": self.support_model_uuid},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class GetEvalTask(BaseModel):
    eval_task_uuid: Union[UUID, None]

    @model_validator(mode="after")
    def check(self: "GetEvalTask") -> "GetEvalTask":
        error_handler = ResponseErrorHandler()

        if self.eval_task_uuid is not None and self.eval_task_uuid.version != 4:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="UUID version 4 expected",
                input={"eval_task_uuid": str(self.eval_task_uuid)},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
