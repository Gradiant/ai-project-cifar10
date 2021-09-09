import json

from stages.data.transform.random_split_classification import (
    random_split_classification,
)


def test_random_split_different_proportions(tmp_path):
    images = [
        {"id": 0, "file_name": "apple.jpg", "category_id": 1},
        {"id": 1, "file_name": "orange.jpg", "category_id": 2},
        {"id": 2, "file_name": "orange2.jpg", "category_id": 2},
        {"id": 3, "file_name": "orange3.jpg", "category_id": 2},
        {"id": 4, "file_name": "orange4.jpg", "category_id": 2},
        {"id": 5, "file_name": "orange5.jpg", "category_id": 2},
        {"id": 6, "file_name": "orange6.jpg", "category_id": 2},
    ]

    categories1 = [{"id": 1, "name": "apple"}, {"id": 2, "name": "orange"}]

    coco_data = dict(images=images, categories=categories1)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(coco_data, f)

    random_split_classification(
        tmp_path / "fruits.json",
        output_matched_file=tmp_path / "output" / "fruits_val_02.json",
        val_proportion=0.2,
        output_unmatched_file=tmp_path / "output" / "fruits_train_02.json",
    )
    random_split_classification(
        tmp_path / "fruits.json",
        output_matched_file=tmp_path / "output" / "fruits_val_08.json",
        val_proportion=0.8,
        output_unmatched_file=tmp_path / "output" / "fruits_train_08.json",
    )

    assert (tmp_path / "output" / "fruits_train_02.json").exists()
    assert (tmp_path / "output" / "fruits_train_08.json").exists()
    assert (tmp_path / "output" / "fruits_val_02.json").exists()
    assert (tmp_path / "output" / "fruits_val_08.json").exists()

    val_02 = json.loads((tmp_path / "output" / "fruits_val_02.json").read_text())
    val_08 = json.loads((tmp_path / "output" / "fruits_val_08.json").read_text())
    assert len(val_08["images"]) > len(val_02["images"])


def test_random_split_empty_val(tmp_path):
    images = [
        {"id": 0, "file_name": "apple.jpg", "category_id": 1},
        {"id": 1, "file_name": "orange.jpg", "category_id": 2},
    ]

    categories1 = [{"id": 1, "name": "apple"}, {"id": 2, "name": "orange"}]

    coco_data = dict(images=images, categories=categories1)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(coco_data, f)

    random_split_classification(
        tmp_path / "fruits.json",
        output_matched_file=tmp_path / "output" / "fruits_val.json",
        val_proportion=0.0,
        output_unmatched_file=tmp_path / "output" / "fruits_train.json",
    )

    assert (tmp_path / "output" / "fruits_train.json").exists()
    assert (tmp_path / "output" / "fruits_val.json").exists()

    train = json.loads((tmp_path / "output" / "fruits_train.json").read_text())
    assert len(train["images"]) == 2
    val = json.loads((tmp_path / "output" / "fruits_val.json").read_text())
    assert len(val["images"]) == 0


def test_random_split_empty_train(tmp_path):
    images = [
        {"id": 0, "file_name": "apple.jpg", "category_id": 1},
        {"id": 1, "file_name": "orange.jpg", "category_id": 2},
    ]

    categories1 = [{"id": 1, "name": "apple"}, {"id": 2, "name": "orange"}]

    coco_data = dict(images=images, categories=categories1)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(coco_data, f)

    random_split_classification(
        tmp_path / "fruits.json",
        output_matched_file=tmp_path / "output" / "fruits_val.json",
        val_proportion=1.0,
        output_unmatched_file=tmp_path / "output" / "fruits_train.json",
    )

    assert (tmp_path / "output" / "fruits_train.json").exists()
    assert (tmp_path / "output" / "fruits_val.json").exists()

    train = json.loads((tmp_path / "output" / "fruits_train.json").read_text())
    assert len(train["images"]) == 0
    val = json.loads((tmp_path / "output" / "fruits_val.json").read_text())
    assert len(val["images"]) == 2
