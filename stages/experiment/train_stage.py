import copy

import fire
import mmcv
from mmcv.utils import import_modules_from_strings
from stages.experiment.utils import get_model_type, get_task_lib


def train_stage(task, config, meta=None, timestamp=None):
    library = get_task_lib(task)
    model_type = get_model_type(task)

    apis, data, models, version = (
        import_modules_from_strings([f"{library}.{module}"])[0]
        for module in ["apis", "datasets", "models", "version"]
    )

    try:
        build_model = getattr(models, f"build_{model_type}")
    except AttributeError:
        build_model = getattr(
            models, "build_classifier"
        )  # mmcls uses build_classifier instead of build_model

    train_model = getattr(apis, f"train_{model_type}")

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    model = build_model(config.model)

    datasets = [data.build_dataset(config.data.train)]
    if len(config.workflow) == 2:
        val_dataset = copy.deepcopy(config.data.val)
        val_dataset.pipeline = config.data.train.pipeline
        datasets.append(data.build_dataset(val_dataset))

    if config.checkpoint_config is not None:
        config.checkpoint_config.meta = {
            f"{library}_version": version,
            "config": config.text,
            "CLASSES": datasets[0].CLASSES,
        }

    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        config,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    fire.Fire(train_stage)
