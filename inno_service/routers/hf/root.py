import json

from fastapi import APIRouter, HTTPException, Response, status

from inno_service.routers.hf import schema, utils
from inno_service.utils.error import ResponseErrorHandler

router = APIRouter(prefix="/hf", tags=["HF"])


@router.post("/add-token/")
async def add_hf_token(request_data: schema.PostAddToken):
    error_handler = ResponseErrorHandler()
    try:
        token = utils.add_token(request_data.hf_token)

    except Exception as e:
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
