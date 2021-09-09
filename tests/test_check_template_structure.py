"""Test suite oriented towards checking if the structure of
a rendered ai-project-template is correct.
"""  # noqa: D205, D400
import glob
from pathlib import Path

import pytest

dataset_types = {
    "classification": "ImageNet",
    "detection": "CocoDataset",
    "segmentation": "CustomDataset",
}

post_gen_model_names = {
    "classification": "resnet_18.py",
    "detection": "retinanet_r50_fpn.py",
    "segmentation": "fpn_r50.py",
}

task = "classification"


@pytest.fixture()
def dataset_type():
    """Pytest fixture which returns the .py dataset filename
    from a rendered template.

    Returns:
        The filename as a string.
    """  # noqa: D205, D400
    ds_filename = glob.glob("configs/datasets/*.py")[0]
    with open(ds_filename, "r") as fp:
        line = fp.readline()
    return line.replace('"', "").split("=")[1].strip()


@pytest.mark.skipif(
    task not in dataset_types.keys(), reason="template isn't rendered yet"
)
def test_check_template_structure(dataset_type):
    """Checks if a rendered template has a correct file structure.

    Args:
        dataset_type: pytest fixture returning the filename
        of the template .py dataset file.
    """
    assert dataset_type == dataset_types[task]
    model_name = Path(glob.glob("configs/models/*.py")[0]).name
    assert model_name == post_gen_model_names[task]

    for fpath in glob.glob("stages/data/explore/*.py"):
        assert task in Path(fpath).name
    assert task in Path(glob.glob("stages/model/explore/*.py")[0]).name

    for fpath in glob.glob("stages/experiment/test_stage*.py"):
        assert task in Path(fpath).name

    for fpath in glob.glob("stages/experiment/post*.py"):
        assert task in Path(fpath).name
