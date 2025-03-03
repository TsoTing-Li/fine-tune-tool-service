import os
from typing import Union

import orjson
from fastapi import HTTPException, status
from pydantic import BaseModel, model_validator

from src.thirdparty.redis.handler import redis_sync
from src.utils.error import ResponseErrorHandler

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")


class PostTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostTrain") -> "PostTrain":
        error_handler = ResponseErrorHandler()

        try:
            if redis_sync.client.hexists("TRAIN", self.train_name):
                raise ValueError("train_name already exists")

        except ValueError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class GetTrain(BaseModel):
    train_name: Union[str, None]

    @model_validator(mode="after")
    def check(self: "GetTrain") -> "GetTrain":
        error_handler = ResponseErrorHandler()

        try:
            if self.train_name and not redis_sync.client.hexists(
                "TRAIN", self.train_name
            ):
                raise KeyError("train_name does not exists")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PutTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PutTrain") -> "PutTrain":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists("TRAIN", self.train_name):
                raise KeyError("train_name does not exists")

            info = redis_sync.client.hget("TRAIN", self.train_name)
            if orjson.loads(info)["container"]["train"]["status"] == "active":
                raise ValueError("train_name is being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
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
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class DelTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "DelTrain") -> "DelTrain":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists("TRAIN", self.train_name):
                raise KeyError("train_name does not exists")

            info = redis_sync.client.hget("TRAIN", self.train_name)
            if orjson.loads(info)["container"]["train"]["status"] == "active":
                raise ValueError("train_name is being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
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
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostStartTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStartTrain") -> "PostStartTrain":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists(
                "TRAIN", self.train_name
            ) or not os.path.exists(os.path.join(SAVE_PATH, self.train_name)):
                raise KeyError("train_name does not exists")

            info = redis_sync.client.hget("TRAIN", self.train_name)
            if orjson.loads(info)["container"]["train"]["status"] == "active":
                raise ValueError("train_name is being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
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
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None

        return self


class PostStopTrain(BaseModel):
    train_name: str

    @model_validator(mode="after")
    def check(self: "PostStopTrain") -> "PostStopTrain":
        error_handler = ResponseErrorHandler()

        try:
            if not redis_sync.client.hexists("TRAIN", self.train_name):
                raise KeyError("train_name does not exists")

            info = redis_sync.client.hget("TRAIN", self.train_name)
            if orjson.loads(info)["container"]["train"]["status"] != "active":
                raise KeyError("train_name is not being executed")

        except KeyError as e:
            error_handler.add(
                type=error_handler.ERR_VALIDATE,
                loc=[error_handler.LOC_BODY],
                msg=f"{e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_handler.errors,
            ) from None

        except Exception as e:
            error_handler.add(
                type=error_handler.ERR_REDIS,
                loc=[error_handler.LOC_DATABASE],
                msg=f"Database error: {e}",
                input={"train_name": self.train_name},
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_handler.errors,
            ) from None
