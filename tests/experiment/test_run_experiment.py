from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pytest
from stages.experiment.run_experiment_mlflow import run_experiment_mlflow


@pytest.mark.parametrize("task", ["detection", "classification", "segmentation"])
@patch("stages.experiment.run_experiment_mlflow._get_task")
@patch("stages.experiment.run_experiment_mlflow.load_modules")
def test_run_experiment(
    mock_load_modules, mock_get_task, task, get_default_configs, patch_mlflow
):
    mock_get_task.return_value = task
    mock_load_modules.return_value = MagicMock(), MagicMock(), MagicMock()
    configs = get_default_configs(task)
    run_experiment_mlflow(**configs)

    artifacts = mlflow.get_artifact_uri()[7:]

    assert len(list(Path(artifacts).glob("**/artifacts/config.py"))) == 1
    assert len(list(Path(artifacts).glob("**/artifacts/experiment.json"))) == 1
