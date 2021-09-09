import json
import pickle
from pathlib import Path
from shutil import copy
from typing import Optional

import fire
import mmcv
import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve


@logger.catch(reraise=True)
def post_train_evaluation(
    config,
    predictions_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> None:

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

    ground_truth_file = getattr(config.data.test, "ann_file")

    y_true = []
    with open(ground_truth_file) as txt_file:
        for line in txt_file:
            y_true.append(int(line.strip("\n").split(" ")[1]))

    predictions = pickle.load(open(predictions_file, "rb"))

    y_pred = []
    for prediction in predictions:
        y_pred.append(int(prediction.argmax()))

    if len(config.CLASSES) > 2:

        f1 = f1_score(y_true, y_pred, average="weighted")

    else:

        f1 = f1_score(y_true, y_pred)

        scores = []
        for prediction in predictions:
            scores.append(prediction[1])
        precision, recall, thresholds = precision_recall_curve(y_true, scores)

        with open(f"{results_dir}/prc.json", "w") as fd:
            json.dump(
                {
                    "prc": [
                        {
                            "precision": float(p),
                            "recall": float(r),
                            "threshold": float(t),
                        }
                        for p, r, t in zip(precision, recall, thresholds)
                    ]
                },
                fd,
            )
        copy(f"{results_dir}/prc.json", f"{output_dir}/prc.json")

    metric_results = {"f1_score": f1}
    mmcv.dump(metric_results, f"{results_dir}/metrics.json")
    copy(f"{results_dir}/metrics.json", f"{output_dir}/metrics.json")

    sns.set(color_codes=True)
    plt.figure(figsize=(20, 15))
    plt.title("Confusion Matrix")
    sns.set(font_scale=1.4)
    cm = confusion_matrix(y_true, y_pred, normalize=None)
    ax = sns.heatmap(cm, annot=True, cmap="YlGnBu", cbar_kws={"label": "Scale"},)
    ax.set_xticklabels(config.CLASSES)
    ax.set_yticklabels(config.CLASSES)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # Save confusion matrix
    np.savetxt(f"{results_dir}/confusion_matrix.txt", cm, fmt="%d", newline="\n")
    copy(f"{results_dir}/confusion_matrix.txt", f"{output_dir}/confusion_matrix.txt")


if __name__ == "__main__":
    fire.Fire(post_train_evaluation)
