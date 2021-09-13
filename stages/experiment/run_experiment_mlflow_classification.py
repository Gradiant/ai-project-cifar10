import json
import tempfile
import time
from os import path as osp
from typing import List, Union

import fire
import mlflow
from loguru import logger
from post_train_evaluation_classification import post_train_evaluation
from pydantic import FilePath
from registry_extras import *  # noqa: F401,F403
from setup_stage_classification import setup_stage
from test_stage_classification import test_stage
from train_stage_classification import train_stage


def log_config_to_mlflow(config, env_info_dict):
    model = f"{config.model.type}-{config.model.backbone.type}"

    if "depth" in config.model.backbone:
        model += f"-{config.model.backbone.depth}"

    mlflow.log_params({"epochs": config.total_epochs, "Model": model})

    for k, v in env_info_dict.items():
        if "PyTorch compiling details" in k:
            # Message too long
            continue
        mlflow.log_param(k.replace(",", "-"), v)

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, "config.py"), "w") as f:
            f.write(config.pretty_text)
            f.seek(0)
            mlflow.log_artifact(f.name)


def run_experiment_mlflow(
    dataset: FilePath,
    model: FilePath,
    optimizer: FilePath,
    scheduler: FilePath,
    gpus: Union[List[int], int] = 0,
):
    with mlflow.start_run():

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        artifact_uri = mlflow.get_artifact_uri()[7:]

        with open("results/experiment.json", "w") as f:
            json.dump({"artifact_uri": artifact_uri, "timestamp": timestamp}, f)

        mlflow.log_artifact("results/experiment.json")

        config, meta, env_info_dict = setup_stage(
            dataset, model, optimizer, scheduler, gpus, artifact_uri, timestamp
        )

        log_config_to_mlflow(config, env_info_dict)

        train_stage(config, meta, timestamp)

        test_stage(config)

        post_train_evaluation(config)


@logger.catch(reraise=True)
def main():
    fire.Fire(run_experiment_mlflow)


if __name__ == "__main__":
    main()
