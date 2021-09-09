from unittest.mock import patch

import pytest
from stages.experiment.setup_stage import build_config
from stages.experiment.train_stage import train_stage
from stages.experiment.utils import get_model_type, get_task_lib


@pytest.mark.parametrize("task", ["detection", "classification", "segmentation"])
def test_run_experiment(task, get_default_configs):
    configs = get_default_configs(task)
    config = build_config(**configs, artifact_uri="")

    lib = get_task_lib(task)
    model_type = get_model_type(task)

    with patch(f"{lib}.apis.train_{model_type}") as mock_train:
        with patch(f"{lib}.datasets.build_dataset") as mock_dataset:
            train_stage(task, config)

    assert mock_train.call_count == 1
    assert mock_dataset.call_count == 1
