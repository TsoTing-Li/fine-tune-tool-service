import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.routers.hf import schema, utils
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger

router = APIRouter(prefix="/hf", tags=["HF"])
security = HTTPBearer()


@router.post("/token/")
async def add_hf_token(request_data: schema.PostAddToken):
    error_handler = ResponseErrorHandler()

    try:
        await utils.call_hf_whoami(hf_token=request_data.hf_token)
        token = utils.add_token(request_data.hf_token)

    except ValueError as e:
        accel_logger.error(f"{e}")
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_BODY],
            msg=f"{e}",
            input={"hf_token": request_data.hf_token},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=error_handler.errors
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=request_data.model_dump(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"hf_token": token}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@router.get("/token/")
def get_hf_token():
    error_handler = ResponseErrorHandler()

    try:
        token = utils.get_token()

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input=dict(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"hf_token": token}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/whoami/")
async def check_hf_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    hf_token = credentials.credentials
    error_handler = ResponseErrorHandler()

    try:
        token_info = await utils.call_hf_whoami(hf_token=hf_token)

    except ValueError as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_HEADERS],
            msg=f"{e}",
            input={},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=error_handler.errors
        ) from None

    except Exception as e:
        accel_logger.error(f"Unexpected error: {e}")
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg="Unexpected error",
            input={},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps(token_info),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
