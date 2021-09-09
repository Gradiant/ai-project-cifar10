import os
from os import path as osp

import torch
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from stages.experiment.utils import get_task_lib


def build_config(dataset, model, runtime, scheduler, artifact_uri, gpus=1):
    config = dict()

    for config_base_file in [dataset, model, runtime, scheduler]:
        config.update(Config._file2dict(config_base_file)[0])

    config["work_dir"] = artifact_uri
    config["gpus"] = gpus
    config["gpu_ids"] = range(gpus)
    config["seed"] = None

    return Config(config)


def set_visible_gpus(gpus):
    if isinstance(gpus, list):
        gpus = ",".join(map(str, gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)


def setup_stage(
    dataset, model, runtime, scheduler, gpus, artifact_uri, timestamp, task
):

    utils = import_modules_from_strings([f"{get_task_lib(task)}.utils"])[0]

    config = build_config(dataset, model, runtime, scheduler, artifact_uri)

    log_file = osp.join(artifact_uri, "{}.log".format(timestamp))
    logger = utils.get_root_logger(log_file=log_file, log_level=config.log_level)

    logger.info("Setting CUDA_VISIBLE_DEVICES to: {}".format(gpus))
    set_visible_gpus(gpus)

    torch.backends.cudnn.benchmark = True

    meta = dict()
    env_info_dict = utils.collect_env()

    env_info = "\n".join([("{}: {}".format(k, v)) for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    meta["env_info"] = env_info
    meta["seed"] = None
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    logger.info("Config:\n{}".format(config.text))

    return config, meta, env_info_dict
