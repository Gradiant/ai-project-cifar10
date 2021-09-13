import copy

import fire
import mmcv
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def train_stage(config, meta=None, timestamp=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    model = build_detector(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )

    datasets = [build_dataset(config.data.train)]
    if len(config.workflow) == 2:
        val_dataset = copy.deepcopy(config.data.val)
        val_dataset.pipeline = config.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if config.checkpoint_config is not None:
        config.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=config.text, CLASSES=datasets[0].CLASSES,
        )

    model.CLASSES = datasets[0].CLASSES
    train_detector(
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
