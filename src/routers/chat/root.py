import json

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from src.routers.chat import schema, utils, validator
from src.thirdparty.redis.handler import redis_async
from src.utils.error import ResponseErrorHandler
from src.utils.logger import accel_logger
from src.utils.utils import generate_uuid

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/stream/start/")
async def start_chat(request_data: schema.PostStartChat):
    error_handler = ResponseErrorHandler()

    try:
        request_id = generate_uuid()
        await redis_async.client.hset("chat_requests", request_id, "processing")

        return StreamingResponse(
            utils.post_openai_chat(
                request_id=request_id,
                model_service=request_data.model_service,
                model_name=request_data.chat_model_name,
                messages=request_data.messages,
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
    validator.PostStopChat(request_id=request_data.request_id)
    error_handler = ResponseErrorHandler()

    try:
        await redis_async.client.hset(
            "chat_requests", request_data.request_id, "cancelled"
        )
        await redis_async.client.publish(
            "chat_requests", f"{request_data.request_id}:cancelled"
        )

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

    finally:
        request_id_exitst = await redis_async.client.hexists(
            "chat_requests", request_data.request_id
        )
        if request_id_exitst:
            await redis_async.client.hdel("chat_requests", request_data.request_id)

    return Response(
        content=json.dumps({"request_id": request_data.request_id}),
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
