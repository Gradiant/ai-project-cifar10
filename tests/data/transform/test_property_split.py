import json

from stages.data.transform.property_split import property_split


def test_simple_val_split(tmp_path):
    images = [{"id": 0, "file_name": "apple.jpg"}, {"id": 1, "file_name": "orange.jpg"}]

    anns1 = [
        {"image_id": 0, "category_id": 1, "id": 0},
        {"image_id": 1, "category_id": 2, "id": 1},
    ]

    categories1 = [{"id": 1, "name": "apple"}, {"id": 2, "name": "orange"}]

    coco_data = dict(images=images, annotations=anns1, categories=categories1)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(coco_data, f)

    property_split(
        tmp_path / "fruits.json",
        output_matched_file=tmp_path / "output" / "fruits_val.json",
        property_name="file_name",
        pattern_to_match="apple",
        output_unmatched_file=tmp_path / "output" / "fruits_train.json",
    )

    assert (tmp_path / "output" / "fruits_train.json").exists()
    assert (tmp_path / "output" / "fruits_val.json").exists()

    train = json.loads((tmp_path / "output" / "fruits_train.json").read_text())
    assert len(train["images"]) == 1
    assert train["images"][0]["file_name"] == "orange.jpg"
    assert len(train["annotations"]) == 1
    val = json.loads((tmp_path / "output" / "fruits_val.json").read_text())
    assert len(val["images"]) == 1
    assert val["images"][0]["file_name"] == "apple.jpg"
    assert len(val["annotations"]) == 1
