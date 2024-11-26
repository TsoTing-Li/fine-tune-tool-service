from fastapi import FastAPI

import inno_service.routers.data.root
import inno_service.routers.deepspeed.root
import inno_service.routers.train.root

inno_api = FastAPI()

inno_api.include_router(inno_service.routers.data.root.router)
inno_api.include_router(inno_service.routers.deepspeed.root.router)
inno_api.include_router(inno_service.routers.train.root.router)


@inno_api.get("/")
def test_api():
    return {"message": "inno api check!"}
