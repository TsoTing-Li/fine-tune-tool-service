from contextlib import asynccontextmanager

import aiofiles
import orjson
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.config.params import COMMON_CONFIG, TASK_CONFIG
from src.routers.main import acceltune_api
from src.schema.eval_tasks import EvalTaskInfo
from src.schema.support_models import SupportModelInfo
from src.thirdparty.redis.handler import redis_async
from src.utils.logger import accel_logger
from src.utils.utils import check_dataset_info_file, generate_uuid


@asynccontextmanager
async def lifespan(app: FastAPI):
    accel_logger.info("Started Service")

    await redis_async.client.delete(TASK_CONFIG.support_model)
    await redis_async.client.delete(TASK_CONFIG.eval_tasks)

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
        uuid = generate_uuid()
        model_name = model["model_name"]
        template = model["template"]
        lora_module = model["lora_module"]
        support_model_info = SupportModelInfo(
            uuid=uuid, name=model_name, template=template, lora_module=lora_module
        ).model_dump()

        await redis_async.client.hset(
            TASK_CONFIG.support_model, uuid, orjson.dumps(support_model_info)
        )

    for task in eval_tasks:
        uuid = generate_uuid()
        task_name = task["task_name"]
        tool_input = task["tool_input"]
        group = task["group"]
        eval_task_info = EvalTaskInfo(
            uuid=uuid, name=task_name, tool_input=tool_input, group=group
        ).model_dump()

        await redis_async.client.hset(
            TASK_CONFIG.eval_tasks, uuid, orjson.dumps(eval_task_info)
        )

    await check_dataset_info_file(
        file_path=f"{COMMON_CONFIG.data_path}/dataset_info.json"
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
