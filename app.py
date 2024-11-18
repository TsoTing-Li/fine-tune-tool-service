# import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from llamafactory.webui.interface import create_ui
from inno_service.routers.main import inno_api

CUSTOM_PATH = "/gradio"

app = FastAPI()

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


# app = gr.mount_gradio_app(app, create_ui(), path=CUSTOM_PATH)
app.mount("/inno", inno_api)
