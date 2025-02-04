from pydantic import BaseModel, ConfigDict


class PostModelServiceAdapterStart(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )  # solve can not start with "model_"
    model_name: str


class PostModelServiceAdapterStop(BaseModel):
    container_name: str
