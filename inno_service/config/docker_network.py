from pydantic import BaseModel


class DockerNetworkConfig(BaseModel):
    network_name: str
