import time

import pytest
from stages.experiment.setup_stage import setup_stage


@pytest.mark.parametrize("task", ["detection", "classification", "segmentation"])
def test_setup_stage_with_default_configs(task, get_default_configs, tmpdir):
    configs = get_default_configs(task)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    config, meta, env_info_dict = setup_stage(
        **configs, gpus=0, artifact_uri=tmpdir, timestamp=timestamp, task=task
    )
