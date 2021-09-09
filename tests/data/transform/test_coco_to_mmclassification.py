import json

import pytest
from stages.data.transform.coco_to_mmclassification import coco_to_mmclassification


def test_coco_to_mmclassification_default(tmp_path):
    coco_data = dict(
        images=[
            {"id": 0, "file_name": "ones.png", "category_id": 0},
            {"id": 1, "file_name": "twos.png", "category_id": 1},
        ],
        categories=[{"id": 0, "name": "ones"}, {"id": 1, "name": "twos"}],
    )
    annotations_file = tmp_path / "annotations_file.json"
    annotations_file.write_text(json.dumps(coco_data))

    coco_to_mmclassification(annotations_file, tmp_path / "output.txt")
    assert (tmp_path / "output.txt").exists()

    output = (tmp_path / "output.txt").read_text().splitlines()
    assert output[0].split(" ")[0] == "ones.png"
    assert output[0].split(" ")[1] == "0"
    assert output[1].split(" ")[0] == "twos.png"
    assert output[1].split(" ")[1] == "1"


def test_coco_to_mmclassification_incorrect_format(tmp_path):
    coco_data = dict(
        images=[
            {"id": 0, "file_name": "ones.png", "category_id": 1},
            {"id": 1, "file_name": "twos.png", "category_id": 2},
        ],
        categories=[{"id": 1, "name": "ones"}, {"id": 2, "name": "twos"}],
    )
    annotations_file = tmp_path / "annotations_file.json"
    annotations_file.write_text(json.dumps(coco_data))

    with pytest.raises(ValueError, match="mmclassification format"):
        coco_to_mmclassification(annotations_file, tmp_path / "output.txt")


def test_coco_to_mmclassification_subtract_one(tmp_path):
    coco_data = dict(
        images=[
            {"id": 0, "file_name": "ones.png", "category_id": 1},
            {"id": 1, "file_name": "twos.png", "category_id": 2},
        ],
        categories=[{"id": 1, "name": "ones"}, {"id": 2, "name": "twos"}],
    )
    annotations_file = tmp_path / "annotations_file.json"
    annotations_file.write_text(json.dumps(coco_data))

    coco_to_mmclassification(
        annotations_file, tmp_path / "output.txt", subtract_one=True
    )
    assert (tmp_path / "output.txt").exists()

    output = (tmp_path / "output.txt").read_text().splitlines()
    assert output[0].split(" ")[0] == "ones.png"
    assert output[0].split(" ")[1] == "0"
    assert output[1].split(" ")[0] == "twos.png"
    assert output[1].split(" ")[1] == "1"
