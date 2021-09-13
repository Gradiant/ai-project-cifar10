from pathlib import Path
from shutil import copy
from typing import Optional

import fire
import mmcv
from loguru import logger
from mmseg.core.evaluation import eval_metrics
from mmseg.datasets import build_dataset


@logger.catch(reraise=True)
def post_train_evaluation(
    config,
    predictions_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    results_dir: str = "results",
) -> None:
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    if predictions_file is None:
        predictions_file = f"{config.work_dir}/test_predictions.pkl"

    if output_dir is None:
        output_dir = config.work_dir

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    predictions = mmcv.load(predictions_file)

    config.data.test.test_mode = True
    dataset = build_dataset(config.data.test)
    gt_seg_maps = dataset.get_gt_seg_maps()

    metrics = eval_metrics(
        predictions,
        gt_seg_maps,
        num_classes=len(config.CLASSES),
        ignore_index=-1,
        metrics=["mIoU", "mDice"],
    )
    metrics_dict = {"mean_acc": metrics["aAcc"]}
    for n, class_name in enumerate(config.CLASSES):
        metrics_dict[f"acc_{class_name}"] = metrics["Acc"][n]
        metrics_dict[f"IoU_{class_name}"] = metrics["IoU"][n]
        metrics_dict[f"Dice_{class_name}"] = metrics["Dice"][n]
    metrics_dict["mIoU"] = metrics["IoU"].mean()
    metrics_dict["mDice"] = metrics["Dice"].mean()

    mmcv.dump(metrics_dict, f"{results_dir}/metrics.json")
    copy(f"{results_dir}/metrics.json", f"{output_dir}/metrics.json")


if __name__ == "__main__":
    fire.Fire(post_train_evaluation)
