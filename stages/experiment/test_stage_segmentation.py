from typing import Optional

import fire
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def test_stage(
    config,
    checkpoint_file: Optional[str] = None,
    output_file: Optional[str] = None,
    output_painted_dir: Optional[str] = None,
) -> None:

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    if checkpoint_file is None:
        checkpoint_file = f"{config.work_dir}/latest.pth"

    if output_file is None:
        output_file = f"{config.work_dir}/test_predictions.pkl"

    if output_painted_dir is None:
        output_painted_dir = f"{config.work_dir}/test_predictions_painted"

    config.model.pretrained = None
    config.data.test.test_mode = True

    dataset = build_dataset(config.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=config.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_segmentor(
        config.model, train_cfg=config.get("train_cfg"), test_cfg=config.get("test_cfg")
    )

    checkpoint = load_checkpoint(model, checkpoint_file, map_location="cpu")

    model.CLASSES = checkpoint["meta"]["CLASSES"]
    model.PALETTE = checkpoint["meta"]["PALETTE"]
    model = MMDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(
        model, data_loader, efficient_test=True, show=False, out_dir=output_painted_dir
    )

    mmcv.dump(outputs, output_file)


if __name__ == "__main__":
    fire.Fire(test_stage)
