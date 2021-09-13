import copy

import fire
import mmcv
from mmcls import __version__
from mmcls.apis import train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier


def train_stage(config, meta=None, timestamp=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    model = build_classifier(config.model)

    datasets = [build_dataset(config.data.train)]
    if len(config.workflow) == 2:
        val_dataset = copy.deepcopy(config.data.val)
        val_dataset.pipeline = config.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if config.checkpoint_config is not None:
        config.checkpoint_config.meta = dict(
            mmcls_version=__version__, config=config.text, CLASSES=datasets[0].CLASSES,
        )

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
