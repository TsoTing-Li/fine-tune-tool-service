import json
import os
from typing import Annotated

from fastapi import APIRouter, Query, Response, status
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse

from inno_service.routers.accelbrain import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler

SAVE_PATH = os.getenv("SAVE_PATH", "/app/saves")
DEPLOY_PATH = os.getenv("EXPORT_PATH", "/app/deploy")

router = APIRouter(prefix="/accelbrain", tags=["Accelbrain"])


@router.post("/deploy/")
async def deploy_accelbrain(request_data: schema.PostDeploy):
    validator.PostDeploy(deploy_name=request_data.deploy_name)
    error_handler = ResponseErrorHandler()
    accelbrain_url = os.getenv("ACCELBRAIN_URL", None)

    try:
        assert accelbrain_url, "'accelbrain_url' not found, save 'accelbrain_url' first"

        return StreamingResponse(
            content=utils.deploy_to_accelbrain_service(
                file_path=os.path.join(SAVE_PATH, request_data.deploy_name, "quantize"),
                model_name=request_data.deploy_name,
                deploy_path=os.path.join(
                    DEPLOY_PATH, f"{request_data.deploy_name}.zip"
                ),
                accelbrain_url=accelbrain_url,
            ),
            status_code=status.HTTP_200_OK,
            media_type="text/event-stream",
        )

    except AssertionError as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_BODY],
            msg=f"{e}",
            input={"accelbrain_url": accelbrain_url},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=error_handler.errors
        ) from None

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"Unexpected error: {e}",
            input={"deploy_name": request_data.deploy_name},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None


@router.get("/health/")
async def check_accelbrain(accelbrain_url: Annotated[str, Query(...)]):
    query_data = schema.GetHealthcheck(accelbrain_url=accelbrain_url)
    error_handler = ResponseErrorHandler()

    try:
        accelbrain_status, accelbrain_status_code = await utils.check_accelbrain_url(
            accelbrain_url=query_data.accelbrain_url
        )

    except ValueError as e:
        error_handler.add(
            type=error_handler.ERR_VALIDATE,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"accelbrain_url": query_data.accelbrain_url},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_handler.errors,
        ) from None

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_QUERY],
            msg=f"{e}",
            input={"accelbrain_url": query_data.accelbrain_url},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"status": accelbrain_status}),
        status_code=accelbrain_status_code,
        media_type="application/json",
    )


@router.post("/save_url/")
async def save_url(request_data: schema.PostSaveurl):
    error_handler = ResponseErrorHandler()

    try:
        accelbrain_url = utils.save_url_in_env(url=request_data.accelbrain_url)

    except Exception as e:
        error_handler.add(
            type=error_handler.ERR_INTERNAL,
            loc=[error_handler.LOC_PROCESS],
            msg=f"{e}",
            input={"accelbrain_url": accelbrain_url},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.errors,
        ) from None

    return Response(
        content=json.dumps({"accelbrain_url": accelbrain_url}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
