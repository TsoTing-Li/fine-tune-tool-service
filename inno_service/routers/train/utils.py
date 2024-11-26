import json
import re
import subprocess

import aiofiles
import yaml

from inno_service.thirdparty.redis.handler import AsyncRedisClient


def yaml_preprocess(data) -> dict:
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
        print("write yaml done")

    except FileNotFoundError:
        raise FileNotFoundError(f"{path} does not exists") from None


parse_patterns = {
    "train_progress": {
        "pattern": re.compile(r"(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[.*?\]"),
        "handler": lambda match, exclude_flag: (
            match.group(0).strip()
            if not exclude_flag
            or "[00:00<?, ?it/s]" not in match.group(0)
            and "[00:00<00:00," not in match.group(0)
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

    return log_info


async def run_train(cmd: str, pub_chan: str):
    process = subprocess.Popen(
        cmd,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )

    try:
        async_redis = AsyncRedisClient()
        exclude_flag = False
        for line in iter(process.stdout.readline, ""):
            log_info = parse(stdout=line, exclude_flag=exclude_flag)
            await async_redis.publish_msg(channel=pub_chan, msg=json.dumps(log_info))

            if "[00:00<?, ?it/s]" in log_info["train_progress"]:
                exclude_flag = True

        process.stdout.close()

        return_code = process.wait()
        await async_redis.publish_msg(channel=pub_chan, msg="FINISHED")
        if return_code != 0:
            print(f"Error occurred: {process.stderr.read()}")
    except Exception as e:
        await async_redis.publish_msg(channel=pub_chan, msg=str(e))
        await async_redis.publish_msg(channel=pub_chan, msg="ERROR")
    finally:
        process.terminate()
