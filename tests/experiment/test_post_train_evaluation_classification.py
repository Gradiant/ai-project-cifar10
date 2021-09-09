import csv
import json
import pickle

import numpy as np
from mmcv.utils import Config
from stages.experiment.post_train_evaluation_classification import post_train_evaluation


def compute_metrics(work_dir, classes, y_true, predictions):
    test = dict(
        type="ImageNet",
        ann_file=f"{work_dir}/ground_truth.json",
        img_prefix=".",
        pipeline=[],
        classes=classes,
    )
    config = Config(dict(CLASSES=classes, work_dir=work_dir, data=dict(test=test)))

    with open(f"{config.work_dir}/test_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    with open(config.data.test["ann_file"], "w", newline="\n") as f:
        gt_writer = csv.writer(f, delimiter=" ")
        for k in range(100):
            gt_writer.writerow(["im" + str(k) + ".png", y_true[k]])

    post_train_evaluation(config, results_dir=f"{work_dir}/results")


def test_binary_classification_metrics(tmpdir):

    # Generate test data
    classes = ["negative", "positive"]

    y_true = [0] * 100
    for k in range(50, 100):
        y_true[k] = 1

    predictions = np.random.rand(100, 2)
    predictions[:, 1] = 1 - predictions[:, 0]

    compute_metrics(tmpdir, classes, y_true, predictions)

    # Perform tests
    prc_filepath = tmpdir / "/results/prc.json"
    metrics_filepath = tmpdir / "results/metrics.json"
    cm_filepath = tmpdir / "results/confusion_matrix.txt"

    assert prc_filepath.exists()
    assert metrics_filepath.exists()
    assert cm_filepath.exists()

    metrics = json.load(open(f"{tmpdir}/results/metrics.json"))
    assert metrics["f1_score"] >= 0
    assert metrics["f1_score"] <= 1

    prc = json.load(open(f"{tmpdir}/results/prc.json"))
    prc = prc["prc"]
    recall = [point["recall"] for point in prc]
    precision = [point["precision"] for point in prc]
    threshold = [point["threshold"] for point in prc]
    assert len(recall) > 2
    assert all([r >= 0 and r <= 1 for r in recall])
    assert all([p >= 0 and p <= 1 for p in precision])
    assert all([th >= 0 and th <= 1 for th in threshold])

    cm = np.loadtxt(cm_filepath)
    assert cm.shape[0] == 2
    assert cm.shape[1] == 2
    assert np.sum(cm[:]) == 100


def test_multiclass_classification_metrics(tmpdir):

    # Generate test data
    classes = ["apple", "orange", "banana"]

    y_true = [0] * 100
    for k in range(50, 75):
        y_true[k] = 1
    for k in range(75, 100):
        y_true[k] = 2

    predictions = np.random.rand(100, 3)
    for row in range(100):
        row_sum = np.sum(predictions[k, :])
        for col in range(3):
            predictions[row, col] = predictions[row, col] / row_sum

    compute_metrics(tmpdir, classes, y_true, predictions)

    # Perform tests
    prc_filepath = tmpdir / "/results/prc.json"
    metrics_filepath = tmpdir / "results/metrics.json"
    cm_filepath = tmpdir / "results/confusion_matrix.txt"

    assert not prc_filepath.exists()
    assert metrics_filepath.exists()
    assert cm_filepath.exists()

    metrics = json.load(open(f"{tmpdir}/results/metrics.json"))
    assert metrics["f1_score"] >= 0
    assert metrics["f1_score"] <= 1

    cm = np.loadtxt(cm_filepath)
    assert cm.shape[0] == 3
    assert cm.shape[1] == 3
    assert np.sum(cm[:]) == 100
