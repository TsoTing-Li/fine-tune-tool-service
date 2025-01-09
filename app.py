from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inno_service.routers.main import inno_api
from inno_service.utils.logger import accel_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    accel_logger.info("Started Service")

    yield

    accel_logger.info("End Service")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_main():
    return {"message": "This is your main app"}


app.mount("/inno", inno_api)
