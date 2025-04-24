from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostStopChat(BaseModel):
    request_id: str

    @model_validator(mode="after")
    def check(self: "PostStopChat") -> "PostStopChat":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists("chat_requests", self.request_id):
                raise KeyError("request_id does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"request_id": self.request_id},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg="Database error",
                input={"request_id": self.request_id},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
