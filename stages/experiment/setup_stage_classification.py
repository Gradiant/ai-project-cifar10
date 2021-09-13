import os
from os import path as osp

import torch
from mmcls.utils import collect_env, get_root_logger
from mmcv import Config

CONFIG_TEMPLATE = dict(
    checkpoint_config=dict(interval=1, max_keep_ckpts=2),
    log_level="INFO",
    log_config=dict(
        interval=50,
        hooks=[
            dict(type="TextLoggerHook"),
            dict(type="MlflowLoggerHook", exp_name="{{cookiecutter.project_slug}}"),
        ],
    ),
    resume_from=None,
    dist_params=dict(backend="nccl"),
)


def build_config(dataset, model, optimizer, scheduler, artifact_uri, gpus=1):
    base_config = dict()

    for config_base_file in [dataset, model, optimizer, scheduler]:
        base_config.update(Config._file2dict(config_base_file)[0])

    config = Config({**base_config, **CONFIG_TEMPLATE})

    config["work_dir"] = artifact_uri
    config["gpus"] = gpus
    config["gpu_ids"] = range(gpus)
    config["seed"] = None

    return config


def set_visible_gpus(gpus):
    if isinstance(gpus, list):
        gpus = ",".join(map(str, gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)


def setup_stage(dataset, model, optimizer, scheduler, gpus, artifact_uri, timestamp):

    config = build_config(dataset, model, optimizer, scheduler, artifact_uri)

    log_file = osp.join(artifact_uri, "{}.log".format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=config.log_level)

    logger.info("Setting CUDA_VISIBLE_DEVICES to: {}".format(gpus))
    set_visible_gpus(gpus)

    torch.backends.cudnn.benchmark = True

    meta = dict()
    env_info_dict = collect_env()

    env_info = "\n".join([("{}: {}".format(k, v)) for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    meta["env_info"] = env_info
    meta["seed"] = None
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    logger.info("Config:\n{}".format(config.text))

    return config, meta, env_info_dict
