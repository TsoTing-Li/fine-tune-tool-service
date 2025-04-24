import re
from typing import Union

from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.utils.error import ResponseErrorHandler


class GetSupportModel(BaseModel):
    base_model: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetSupportModel") -> "GetSupportModel":
        error_handler = ResponseErrorHandler()

        if (
            self.base_model is not None
            and bool(re.search(r"[^a-zA-Z0-9_\-\./]+", self.base_model)) is True
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="base_model contain invalid characters",
                input={"base_model": self.base_model},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self


class GetEvalTask(BaseModel):
    eval_task: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetEvalTask") -> "GetEvalTask":
        error_handler = ResponseErrorHandler()

        if (
            self.eval_task is not None
            and bool(re.search(r"[^a-zA-Z0-9_\-\s\./]+", self.eval_task)) is True
        ):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="eval_task contain invalid characters",
                input={"eval_task": self.eval_task},
            )

        if error_handler.errors != []:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_handler.errors,
            )

        return self
