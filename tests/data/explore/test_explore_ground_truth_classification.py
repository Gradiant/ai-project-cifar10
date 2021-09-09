import json

import pandas as pd
from stages.data.explore.explore_ground_truth_classification import (
    explore_ground_truth_classification,
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

    ground_truth = dict(images=images, categories=categories1)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(ground_truth, f)

    explore_ground_truth_classification(tmp_path / "fruits.json", tmp_path / "output")

    assert (tmp_path / "output" / "fruits.csv").exists()
    assert (tmp_path / "output" / "category_distribution.html").exists()
    assert (tmp_path / "output" / "image_shape_distribution.html").exists()
    ground_truth_df = pd.read_csv(tmp_path / "output" / "fruits.csv")
    assert len(ground_truth_df) == len(images)
    assert "category" in ground_truth_df.columns
    assert "width" in ground_truth_df.columns
    assert "height" in ground_truth_df.columns


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

    ground_truth = dict(images=images, categories=categories, colors=colors)

    with open(tmp_path / "fruits.json", "w") as f:
        json.dump(ground_truth, f)

    explore_ground_truth_classification(
        tmp_path / "fruits.json",
        tmp_path / "output",
        categories="colors",
        category_id="color_id",
    )

    assert (tmp_path / "output" / "fruits.csv").exists()
    assert (tmp_path / "output" / "category_distribution.html").exists()
    assert (tmp_path / "output" / "image_shape_distribution.html").exists()
    ground_truth_df = pd.read_csv(tmp_path / "output" / "fruits.csv")
    assert len(ground_truth_df) == len(images)
    assert "category" in ground_truth_df.columns
    assert "width" in ground_truth_df.columns
    assert "height" in ground_truth_df.columns
    assert "green" in ground_truth_df["category"].values
    assert "yellow" in ground_truth_df["category"].values
