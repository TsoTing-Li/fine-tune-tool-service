from fastapi import FastAPI, Response, status
from fastapi.responses import PlainTextResponse

import src.routers.accelbrain.root
import src.routers.chat.root
import src.routers.data.root
import src.routers.deepspeed.root
import src.routers.evaluate.root
import src.routers.hf.root
import src.routers.infer_backend.root
import src.routers.info.root
import src.routers.merge.root
import src.routers.ollama.root
import src.routers.quantize.root
import src.routers.train.root
import src.routers.vllm.root
import src.routers.ws.root

acceltune_api = FastAPI()

acceltune_api.include_router(src.routers.data.root.router)
acceltune_api.include_router(src.routers.deepspeed.root.router)
acceltune_api.include_router(src.routers.train.root.router)
acceltune_api.include_router(src.routers.quantize.root.router)
acceltune_api.include_router(src.routers.ws.root.router)
acceltune_api.include_router(src.routers.vllm.root.router)
acceltune_api.include_router(src.routers.chat.root.router)
acceltune_api.include_router(src.routers.evaluate.root.router)
acceltune_api.include_router(src.routers.accelbrain.root.router)
acceltune_api.include_router(src.routers.ollama.root.router)
acceltune_api.include_router(src.routers.infer_backend.root.router)
acceltune_api.include_router(src.routers.hf.root.router)
acceltune_api.include_router(src.routers.info.root.router)
acceltune_api.include_router(src.routers.merge.root.router)


@acceltune_api.get("/health/", tags=["Health"], response_class=PlainTextResponse)
def health_check():
    return Response(content="", status_code=status.HTTP_200_OK, media_type="text/plain")
