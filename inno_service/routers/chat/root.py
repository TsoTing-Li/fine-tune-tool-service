import json

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from inno_service.routers.chat import schema, utils, validator
from inno_service.utils.error import ResponseErrorHandler
from inno_service.utils.utils import generate_uuid

router = APIRouter(prefix="/chat", tags=["Chat"])

active_requests = dict()


@router.post("/stream/start/")
async def start_chat(request_data: schema.PostStartChat):
    error_handler = ResponseErrorHandler()

    try:
        request_id = generate_uuid()
        active_requests[request_id] = "processing"

        return StreamingResponse(
            utils.post_openai_chat(
                request_id=request_id,
                model_server=request_data.model_service,
                model_name=request_data.chat_model_name,
                messages=request_data.messages,
                active_requests=active_requests,
            ),
            status_code=status.HTTP_200_OK,
            media_type="text/event-stream",
        )

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


@router.post("/stream/stop/")
async def stop_chat(request_data: schema.PostStopChat):
    validator.PostStopChat(
        request_id=request_data.request_id, active_requests=active_requests
    )
    error_handler = ResponseErrorHandler()

    try:
        active_requests[request_data.request_id] = "cancelled"

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

    finally:
        if active_requests.get(request_data.request_id):
            active_requests.pop(request_data.request_id, None)

    return Response(
        content=json.dumps({"request_id": request_data.request_id}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
