from typing import Literal, Union

from pydantic import BaseModel


class CPUTemplate(BaseModel):
    usage: Union[float, Literal["N/A"]] = "N/A"
    avg_temp: Union[float, Literal["N/A"]] = "N/A"


class GPUTemplate(BaseModel):
    device: str = "N/A"
    usage: Union[float, Literal["N/A"]] = "N/A"
    used: Union[float, Literal["N/A"]] = "N/A"
    total: Union[float, Literal["N/A"]] = "N/A"
    temperature: Union[float, Literal["N/A"]] = "N/A"
    gpu_utilization: Union[float, Literal["N/A"]] = "N/A"
    vram_utilization: Union[float, Literal["N/A"]] = "N/A"


class DiskTemplate(BaseModel):
    usage: Union[float, Literal["N/A"]] = "N/A"
    used: Union[float, Literal["N/A"]] = "N/A"
    total: Union[float, Literal["N/A"]] = "N/A"


class MemoryTemplate(BaseModel):
    usage: Union[float, Literal["N/A"]] = "N/A"
    used: Union[float, Literal["N/A"]] = "N/A"
    total: Union[float, Literal["N/A"]] = "N/A"
