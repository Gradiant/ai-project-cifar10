from pathlib import Path
from shutil import copy
from typing import Callable, Dict, List, Optional

import fire
import mmcv
from loguru import logger
from mmdet.core.evaluation import eval_map
from mmdet.datasets import build_dataset


def get_f1_score(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


@logger.catch(reraise=True)
def post_train_evaluation(
    config,
    predictions_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    iou_threshold: float = 0.5,
    scale_ranges: Optional[List[tuple]] = None,
    tpfp_fn: Optional[Callable] = None,
    nproc: Optional[int] = 4,
) -> Dict:

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)

    if predictions_file is None:
        predictions_file = f"{config.work_dir}/test_predictions.pkl"

    if output_dir is None:
        output_dir = config.work_dir

    if results_dir is None:
        results_dir = "results"
    else:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    predictions = mmcv.load(predictions_file)

    config.data.test.test_mode = True
    dataset = build_dataset(config.data.test)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]

    mean_ap, eval_results = eval_map(
        predictions,
        annotations,
        scale_ranges=scale_ranges,
        iou_thr=iou_threshold,
        dataset=config.CLASSES,
        tpfp_fn=tpfp_fn,
        nproc=nproc,
    )

    metrics = dict(total_tp=0, total_fp=0, total_gts=0, mean_ap=mean_ap)

    for category_id, result in enumerate(eval_results):
        category = config.CLASSES[category_id]

        tp, fp, recall, precision, f1_score = 0, 0, 0, 0, 0

        if result["ap"] > 0.0:

            recall = result["recall"][-1]
            precision = result["precision"][-1]
            tp = int(result["num_gts"] * recall)
            fp = int((tp / precision) - tp)
            f1_score = get_f1_score(precision, recall)

        metrics[category] = dict(
            precision=precision,
            recall=recall,
            tp=tp,
            fp=fp,
            ap=result["ap"],
            f1_score=f1_score,
        )

        metrics["total_tp"] += tp
        metrics["total_fp"] += fp
        metrics["total_gts"] += result["num_gts"]

    metrics["mean_precision"] = metrics["total_tp"] / metrics["total_gts"]

    total_positives = metrics["total_tp"] + metrics["total_fp"]

    if total_positives != 0:
        metrics["mean_recall"] = metrics["total_tp"] / (total_positives)
    else:
        metrics["mean_recall"] = 0

    metrics["mean_f1"] = get_f1_score(metrics["mean_precision"], metrics["mean_recall"])

    mmcv.dump(metrics, f"{results_dir}/metrics.json")
    copy(f"{results_dir}/metrics.json", f"{output_dir}/metrics.json")

    return metrics


if __name__ == "__main__":
    fire.Fire(post_train_evaluation)
