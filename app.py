from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from inno_service.routers.main import acceltune_api
from inno_service.thirdparty import redis
from inno_service.utils.logger import accel_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    accel_logger.info("Started Service")

    yield

    await redis.handler.redis_async.aclose()
    accel_logger.info("End Service")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health/")
def health_check():
    return Response(content="", status_code=status.HTTP_200_OK, media_type="text/plain")


app.mount("/acceltune", acceltune_api)
