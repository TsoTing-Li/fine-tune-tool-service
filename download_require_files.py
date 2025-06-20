import asyncio
import json
import os
from pathlib import Path

import anyio
from datasets import load_dataset
from huggingface_hub import snapshot_download


async def get_support_model() -> list:
    support_model_static_file = Path.cwd() / "static/support_model.json"
    async with await anyio.open_file(
        support_model_static_file, mode="r", encoding="utf-8"
    ) as f:
        content = await f.read()

    support_model_data = json.loads(content)
    return support_model_data


async def get_eval_tasks() -> list:
    eval_tasks_static_file = Path.cwd() / "static/eval_tasks.json"
    async with await anyio.open_file(
        eval_tasks_static_file, mode="r", encoding="utf-8"
    ) as f:
        content = await f.read()

    eval_tasks_data = json.loads(content)
    return eval_tasks_data


def download_model(model_list: list) -> None:
    hf_token = os.getenv("HF_TOKEN", "")
    for model_info in model_list:
        model_name = model_info["model_name"]
        print(f"Downloading model {model_name}")
        snapshot_download(repo_id=model_name, token=hf_token)
        print("Model downloaded.")


def download_dataset(dataset_list: list) -> None:
    for dataset_info in dataset_list:
        dataset_name = dataset_info["tool_input"]
        print(f"Downloading dataset {dataset_name}")
        load_dataset(dataset_name, token=os.getenv("HF_TOKEN", ""))
        print("Dataset downloaded.")


async def main() -> None:
    support_model_list = await get_support_model()
    eval_tasks_list = await get_eval_tasks()
    loop = asyncio.get_event_loop()
    model_task = loop.run_in_executor(
        None, download_model(model_list=support_model_list)
    )
    dataset_task = loop.run_in_executor(
        None, download_dataset(dataset_list=eval_tasks_list)
    )

    await asyncio.gather(model_task, dataset_task)


if __name__ == "__main__":
    asyncio.run(main())
