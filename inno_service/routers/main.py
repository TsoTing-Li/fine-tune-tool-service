from fastapi import FastAPI, Response, status

import inno_service.routers.accelbrain.root
import inno_service.routers.chat.root
import inno_service.routers.data.root
import inno_service.routers.deepspeed.root
import inno_service.routers.evaluate.root
import inno_service.routers.hf.root
import inno_service.routers.merge.root
import inno_service.routers.model_service_adapter.root
import inno_service.routers.ollama.root
import inno_service.routers.quantize.root
import inno_service.routers.train.root
import inno_service.routers.vllm.root
import inno_service.routers.ws.root

acceltune_api = FastAPI()

acceltune_api.include_router(inno_service.routers.data.root.router)
acceltune_api.include_router(inno_service.routers.deepspeed.root.router)
acceltune_api.include_router(inno_service.routers.train.root.router)
acceltune_api.include_router(inno_service.routers.quantize.root.router)
acceltune_api.include_router(inno_service.routers.ws.root.router)
acceltune_api.include_router(inno_service.routers.vllm.root.router)
acceltune_api.include_router(inno_service.routers.merge.root.router)
acceltune_api.include_router(inno_service.routers.chat.root.router)
acceltune_api.include_router(inno_service.routers.evaluate.root.router)
acceltune_api.include_router(inno_service.routers.accelbrain.root.router)
acceltune_api.include_router(inno_service.routers.ollama.root.router)
acceltune_api.include_router(inno_service.routers.model_service_adapter.root.router)
acceltune_api.include_router(inno_service.routers.hf.root.router)


@acceltune_api.get("/health/", tags=["Health"])
def health_check():
    return Response(content="", status_code=status.HTTP_200_OK, media_type="text/plain")
