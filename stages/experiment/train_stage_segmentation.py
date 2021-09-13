import copy

import fire
import mmcv
from mmseg import __version__
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


def train_stage(config, meta=None, timestamp=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    model = build_segmentor(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )

    datasets = [build_dataset(config.data.train)]
    if len(config.workflow) == 2:
        val_dataset = copy.deepcopy(config.data.val)
        val_dataset.pipeline = config.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if config.checkpoint_config is not None:
        config.checkpoint_config.meta = dict(
            mmseg_version=__version__,
            config=config.text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE,
        )

    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
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
