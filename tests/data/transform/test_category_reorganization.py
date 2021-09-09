import json
from pathlib import Path

from stages.data.transform.category_reorganization import category_reorganization


def test_merge_two_categories(tmpdir):
    tmpdir = Path(tmpdir)
    images = [{"id": 0, "file_name": "0.jpg"}, {"id": 1, "file_name": "1.jpg"}]

    anns1 = [
        {"image_id": 0, "category_id": 1, "id": 0},
        {"image_id": 1, "category_id": 2, "id": 1},
    ]

    categories1 = [{"id": 1, "name": "apple"}, {"id": 2, "name": "orange"}]

    coco_data = dict(images=images, annotations=anns1, categories=categories1)

    with open(tmpdir / "fruits.json", "w") as f:
        json.dump(coco_data, f)

    result = category_reorganization(
        tmpdir / "fruits.json", ["apple", "orange"], "fruit", tmpdir / "output.json"
    )

    category_names = [x["name"] for x in result["categories"]]
    assert "apple" not in category_names
    assert "orange" not in category_names
    assert "fruit" in category_names


def test_merge_into_existing(tmpdir):
    tmpdir = Path(tmpdir)
    images = [{"id": 0, "file_name": "0.jpg"}, {"id": 1, "file_name": "1.jpg"}]

    anns1 = [
        {"image_id": 0, "dog_id": 1, "id": 0},
        {"image_id": 1, "dog_id": 2, "id": 1},
    ]

    categories1 = [{"id": 1, "name": "dog"}, {"id": 2, "name": "huskey"}]

    coco_data = dict(images=images, annotations=anns1, categories=categories1)

    with open(tmpdir / "dogs.json", "w") as f:
        json.dump(coco_data, f)

    result = category_reorganization(
        tmpdir / "dogs.json",
        ["dog", "huskey"],
        "dog",
        tmpdir / "output.json",
        category_id="dog_id",
    )

    category_names = [x["name"] for x in result["categories"]]
    assert "huskey" not in category_names
    assert "dog" in category_names
