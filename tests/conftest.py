import glob
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
import yaml


@pytest.fixture()
def get_default_configs():
    def _get_configs(task):
        configs = {}
        for config_file in glob.glob("configs/**/*"):
            parent_dir = Path(config_file).parent
            if len(list(parent_dir.iterdir())) == 1 or task in config_file:
                configs[parent_dir.name[:-1]] = config_file
        return configs

    return _get_configs


@pytest.fixture()
def patch_mlflow(tmpdir):
    """Sets mlflow artifact to a tmp directory for all tests"""
    mlruns = tmpdir.mkdir("mlruns")

    mlflow.set_tracking_uri(str(mlruns))
    artifact_location = mlruns.mkdir("0")
    meta = dict(
        artifact_location=str(artifact_location),
        experiment_id=0,
        lifecycle_stage="active",
        name="Default",
    )

    with open(artifact_location.join("meta.yaml"), "w") as f:
        yaml.dump(meta, f)

    # Needed to force mlflow.get_artifact_uri() return tmpdir over all tests
    with patch(
        "mlflow.get_artifact_uri", return_value=f"file://{artifact_location}"
    ) as _fixture:
        yield _fixture
