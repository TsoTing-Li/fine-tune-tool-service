import asyncio
import json
from pathlib import Path
from typing import Dict

import httpx

try:
    import tomllib

    _is_stdlib = True
except ModuleNotFoundError:
    import toml as tomllib

    _is_stdlib = False


async def get_all_image_name(
    docker_username: str,
    docker_base_repo: str,
    micro_service_dict: Dict[str, Dict[str, str]],
    shared_service_dict: Dict[str, Dict[str, str]],
    thirdparty_service_dict: Dict[str, Dict[str, str]],
    frontend_dict: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    all_image_name: Dict[str, Dict[str, str]] = dict()

    for name, info in micro_service_dict.items():
        all_image_name[name] = {
            "image_name": f"{docker_username}/{docker_base_repo}-{info['name']}",
            "tag": info["version"],
        }

    for name, info in shared_service_dict.items():
        all_image_name[name] = {
            "image_name": f"{docker_username}/{info['name']}",
            "tag": info["version"],
        }

    for name, info in thirdparty_service_dict.items():
        all_image_name[name] = {"image_name": info["name"], "tag": info["version"]}

    website_name = frontend_dict["name"]
    website_version = frontend_dict["version"]
    all_image_name[website_name] = {
        "image_name": f"{docker_username}/{docker_base_repo}-{website_name}",
        "tag": website_version,
    }

    return all_image_name


async def pull_docker_image(
    aclient: httpx.AsyncClient, image_name: str, tag: str
) -> None:
    async with aclient.stream(
        "POST",
        "http://docker/images/create",
        params={"fromImage": image_name, "tag": tag},
    ) as response:
        async for line in response.aiter_lines():
            if line.strip():
                try:
                    progress_info = json.loads(line)
                    print(
                        progress_info.get("status"),
                        progress_info.get("progressDetail", ""),
                        progress_info.get("id", ""),
                    )
                except json.JSONDecodeError:
                    print("Non-JSON line:", line)


async def check_docker_image(
    aclient: httpx.AsyncClient, image_name: str, tag: str
) -> bool:
    response = await aclient.get(f"http://docker/images/{image_name}:{tag}/json")

    if response.status_code == 200:
        print(f"Check docker image {image_name}:{tag} --> exists")
        return True
    else:
        print(f"Check docker image {image_name}:{tag} --> not exists")
        return False


async def process_image(
    aclient: httpx.AsyncClient, sem: asyncio.Semaphore, image_name: str, tag: str
):
    async with sem:
        exists = await check_docker_image(
            aclient=aclient, image_name=image_name, tag=tag
        )
        if not exists:
            await pull_docker_image(aclient=aclient, image_name=image_name, tag=tag)


async def main():
    cwd = Path.cwd()
    pyproject_file = cwd / "pyproject.toml"
    version_file = cwd / "version.toml"

    if not pyproject_file.exists():
        raise FileNotFoundError("pyproject.toml not found")

    if not version_file.exists():
        raise FileNotFoundError("version.toml not found")

    mode = "rb" if _is_stdlib else "r"
    with pyproject_file.open(mode) as f:
        pyproject_file_content = tomllib.load(f)

    project_version = pyproject_file_content["project"]["version"]

    with version_file.open(mode) as f:
        version_file_content = tomllib.load(f)

    assert project_version == version_file_content["project"]["version"], (
        "project version does not match env version"
    )

    all_images = await get_all_image_name(
        docker_username=version_file_content["docker"]["info"]["username"],
        docker_base_repo=version_file_content["docker"]["info"]["base_repo"],
        micro_service_dict=version_file_content["services"],
        shared_service_dict=version_file_content["shared"],
        thirdparty_service_dict=version_file_content["thirdparty"],
        frontend_dict=version_file_content["frontend"],
    )

    MAX_CONCURRENT_PULLS = len(all_images)
    transport = httpx.AsyncHTTPTransport(uds="/var/run/docker.sock")
    sem = asyncio.Semaphore(MAX_CONCURRENT_PULLS)

    async with httpx.AsyncClient(transport=transport, timeout=None) as aclient:
        tasks = [
            process_image(
                aclient=aclient,
                sem=sem,
                image_name=image_info["image_name"],
                tag=image_info["tag"],
            )
            for image_info in all_images.values()
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
