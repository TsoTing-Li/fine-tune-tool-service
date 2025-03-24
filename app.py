from contextlib import asynccontextmanager

import aiofiles
import orjson
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.config.params import COMMON_CONFIG, TASK_CONFIG
from src.routers.main import acceltune_api
from src.thirdparty.redis.handler import redis_async
from src.utils.logger import accel_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    accel_logger.info("Started Service")

    async with aiofiles.open(
        f"{COMMON_CONFIG.workspace_path}/static/support_model.json"
    ) as support_model_file:
        support_model_content = await support_model_file.read()
        support_model = orjson.loads(support_model_content)

    async with aiofiles.open(
        f"{COMMON_CONFIG.workspace_path}/static/eval_tasks.json"
    ) as eval_tasks_file:
        eval_tasks_content = await eval_tasks_file.read()
        eval_tasks = orjson.loads(eval_tasks_content)

    for model in support_model:
        await redis_async.client.hset(
            TASK_CONFIG.support_model, model["model_name"], orjson.dumps(model)
        )

    for task in eval_tasks:
        await redis_async.client.hset(
            TASK_CONFIG.eval_tasks, task["task_name"], orjson.dumps(task)
        )

    yield

    await redis_async.aclose()
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
