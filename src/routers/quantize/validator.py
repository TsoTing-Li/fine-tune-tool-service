import orjson
from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.config.params import TASK_CONFIG
from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler


class PostStartQuantize(BaseModel):
    quantize_name: str

    @model_validator(mode="after")
    def check(self: "PostStartQuantize") -> "PostStartQuantize":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.quantize_name)

            if not info:
                raise KeyError("quantize_name does not exists")

            if orjson.loads(info)["container"]["quantize"]["status"] == "active":
                raise ValueError("quantize_name is being quantized")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"quantize_name": self.quantize_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"quantize_name": self.quantize_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"quantize_name": self.quantize_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostStopQuantize(BaseModel):
    quantize_name: str

    @model_validator(mode="after")
    def check(self: "PostStopQuantize") -> "PostStopQuantize":
        error_handler = ResponseErrorHandler()

        try:
            info = redis_sync.client.hget(TASK_CONFIG.train, self.quantize_name)

            if not info:
                raise KeyError("quantize_name does not exists")

            if orjson.loads(info)["container"]["quantize"]["status"] != "active":
                raise KeyError("quantize_name is not being quantized")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"quantize_name": self.quantize_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=error_handler.errors
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"quantize_name": self.quantize_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self
