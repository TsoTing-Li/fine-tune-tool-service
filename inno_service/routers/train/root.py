import json
import os

from fastapi import APIRouter, BackgroundTasks, Response, status

from inno_service.routers.train import schema, utils
from inno_service.utils.utils import get_current_time

SAVE_PATH = "/app/saves"

router = APIRouter(prefix="/train")


@router.post("/", tags=["Train"])
async def post_train(background_task: BackgroundTasks, request_data: schema.PostTrain):
    if not request_data.train_name:
        train_name = get_current_time()
    else:
        train_name = request_data.train_name

    train_args = utils.basemodel2dict(data=request_data.train_args)
    train_args["output_dir"] = os.path.join(SAVE_PATH, train_name, train_args["finetuning_type"])
    train_args["eval_steps"] = train_args["save_steps"]
    yaml_path = os.path.join(SAVE_PATH, train_name, f"{train_name}.yaml")
    await utils.write_train_yaml(path=yaml_path, data=train_args)
    background_task.add_task(
        utils.run_train, f"llamafactory-cli train {yaml_path}", train_name
    )

    return Response(
        content=json.dumps({"train_name": train_name}),
        status_code=status.HTTP_201_CREATED,
        media_type="application/json",
    )
