import json
import os
import re
from typing import Literal

import aiofiles
import httpx
import yaml


def basemodel2dict(data) -> dict:
    train_args = {
        "model_name_or_path": data.model_name_or_path,
        **data.method.model_dump(),
        **data.dataset.model_dump(),
        **data.output.model_dump(),
        **data.params.model_dump(),
        **data.eval.model_dump(),
    }

    return train_args


async def write_train_yaml(path: str, data: dict):
    try:
        yaml_content = yaml.dump(data, default_flow_style=False)

        async with aiofiles.open(path, "w") as af:
            await af.write(yaml_content)

    except FileNotFoundError:
        raise FileNotFoundError(f"{path} does not exists") from None


parse_patterns = {
    "convert_progress": {
        "pattern": re.compile(
            r"^Converting format of dataset \(num_proc=\d+\):\s+\d+%\|.*?\|\s*(\d+/\d+\s\[.*?\])"
        ),
        "handler": lambda match: match.group(1).strip(),
    },
    "run_tokenizer_progress": {
        "pattern": re.compile(
            r"^Running tokenizer on dataset \(num_proc=\d+\):\s+\d+%\|.*?\|\s*(\d+/\d+\s\[.*?\])"
        ),
        "handler": lambda match: match.group(1).strip(),
    },
    "train_progress": {
        "pattern": re.compile(r"^\d+%\|.*?\|\s*(\d+/\d+\s\[.*?\])"),
        "handler": lambda match, exclude_flag: (
            match.group(1).strip()
            if not exclude_flag
            or "[00:00<?, ?it/s]" not in match.group(1)
            and "[00:00<00:00," not in match.group(1)
            else ""
        ),
    },
    "train_loss": {
        "pattern": re.compile(r"{'loss':.*?}"),
        "handler": lambda match: match.group(0).strip(),
    },
    "eval_loss": {
        "pattern": re.compile(r"{'eval_loss':.*?}"),
        "handler": lambda match: match.group(0).strip(),
    },
}


def parse(stdout: str, exclude_flag: bool) -> dict:
    log_info = dict()
    for key, value in parse_patterns.items():
        match = value["pattern"].search(stdout)
        if match:
            if key == "train_progress":
                log_info[key] = value["handler"](match, exclude_flag)
            else:
                log_info[key] = value["handler"](match)
        else:
            log_info[key] = ""

    log_info["ori"] = stdout

    return log_info


async def run_train(base_url: str, image_name: str, cmd: list, train_name: str):
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        response = await aclient.get(
            f"{base_url}/json", params={"filters": json.dumps({"name": [train_name]})}
        )
    home_path = os.environ["HOME_PATH"]
    root_path = os.environ["ROOT_PATH"]
    params = {"name": train_name}
    data = {
        "User": "root",
        "Image": image_name,
        "HostConfig": {
            "IpcMode": "host",
            "DeviceRequests": [
                {"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}
            ],
            "Binds": [
                f"{home_path}/.cache/huggingface:/root/.cache/huggingface:rw",
                f"{root_path}/data:/app/data:rw",
                f"{root_path}/saves/{train_name}:/app/saves/{train_name}:rw",
            ],
        },
        "Cmd": cmd,
    }

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        response = await aclient.post(f"{base_url}/create", json=data, params=params)

        if response.status_code == 201:
            container_id = response.json()["Id"]

            async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
                response = await aclient.post(f"{base_url}/{container_id}/start")

                if response.status_code == 204:
                    print("Container started")
                else:
                    print(f"Container started failed: {response.status_code}")

        else:
            print(f"{response.status_code}\n{response.text}")


async def stop_train(
    base_url: str,
    container_name_or_id: str,
    signal: Literal["SIGINT", "SIGTERM", "SIGKILL"] = "SIGTERM",
    wait_sec: int = 10,
):
    params = {"signal": signal, "t": wait_sec}

    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        response = await aclient.post(
            f"{base_url}/{container_name_or_id}/stop", params=params
        )

        if response.status_code == 204:
            print("Container stop")
        else:
            print(f"Container stop failed: {response.status_code}\n{response.text}")
