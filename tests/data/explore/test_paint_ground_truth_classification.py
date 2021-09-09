import json

import numpy as np
from PIL import Image
from stages.data.explore.paint_ground_truth_classification import (
    paint_ground_truth_classification,
)


def test_default_categories(tmp_path):
    images = [
        {
            "id": 0,
            "file_name": "apple_0.jpg",
            "category_id": 1,
            "width": 10,
            "height": 10,
        },
        {
            "id": 1,
            "file_name": "orange_0.jpg",
            "category_id": 2,
            "width": 10,
            "height": 10,
        },
    ]
    categories1 = [{"id": 1, "name": "apple"}, {"id": 2, "name": "orange"}]
    image_folder = tmp_path / "images"
    image_folder.mkdir()

    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
        image_folder / "apple_0.jpg"
    )
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
        image_folder / "orange_0.jpg"
    )

    ground_truth = dict(images=images, categories=categories1)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(ground_truth, f)

    paint_ground_truth_classification(
        tmp_path / "fruits.json", tmp_path / "images", tmp_path / "output"
    )

    assert (tmp_path / "output" / "apple_0.html").exists()
    assert (tmp_path / "output" / "orange_0.html").exists()


def test_custom_categories(tmp_path):
    images = [
        {
            "id": 0,
            "file_name": "apple_0.jpg",
            "category_id": 1,
            "color_id": 1,
            "width": 10,
            "height": 10,
        },
        {
            "id": 1,
            "file_name": "banana_0.jpg",
            "category_id": 2,
            "color_id": 2,
            "width": 10,
            "height": 10,
        },
    ]

    categories = [{"id": 1, "name": "apple"}, {"id": 2, "name": "banana"}]
    colors = [{"id": 1, "name": "green"}, {"id": 2, "name": "yellow"}]

    image_folder = tmp_path / "images"
    image_folder.mkdir()

    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
        image_folder / "apple_0.jpg"
    )
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
        image_folder / "banana_0.jpg"
    )

    ground_truth = dict(images=images, categories=categories, colors=colors)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(ground_truth, f)

    paint_ground_truth_classification(
        tmp_path / "fruits.json",
        tmp_path / "images",
        tmp_path / "output",
        categories="colors",
        category_id="color_id",
    )

    assert (tmp_path / "output" / "apple_0.html").exists()
    assert (tmp_path / "output" / "banana_0.html").exists()
