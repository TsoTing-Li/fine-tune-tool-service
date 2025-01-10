from fastapi import FastAPI

import inno_service.routers.chat.root
import inno_service.routers.data.root
import inno_service.routers.deepspeed.root
import inno_service.routers.merge.root
import inno_service.routers.quantize.root
import inno_service.routers.train.root
import inno_service.routers.vllm.root
import inno_service.routers.ws.root

inno_api = FastAPI()

inno_api.include_router(inno_service.routers.data.root.router)
inno_api.include_router(inno_service.routers.deepspeed.root.router)
inno_api.include_router(inno_service.routers.train.root.router)
inno_api.include_router(inno_service.routers.quantize.root.router)
inno_api.include_router(inno_service.routers.ws.root.router)
inno_api.include_router(inno_service.routers.vllm.root.router)
inno_api.include_router(inno_service.routers.merge.root.router)
inno_api.include_router(inno_service.routers.chat.root.router)


@inno_api.get("/")
def test_api():
    return {"message": "inno api check!"}
