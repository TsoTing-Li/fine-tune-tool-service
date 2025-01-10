from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, model_validator

from inno_service.utils.error import ResponseErrorHandler


class PostStopChat(BaseModel):
    request_id: str
    active_requests: dict

    @model_validator(mode="after")
    def check(self: "PostStopChat") -> "PostStopChat":
        error_handler = ResponseErrorHandler()

        if not self.active_requests.get(self.request_id):
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg="'request_id' does not exists",
                input={"request_id": self.request_id},
            )

        if error_handler.errors != []:
            raise RequestValidationError(error_handler.errors)

        return self
