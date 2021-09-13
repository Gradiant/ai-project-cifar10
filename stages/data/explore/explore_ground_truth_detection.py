import json
from collections import defaultdict
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def explore_ground_truth_detection(
    ground_truth_file: str,
    output_folder: str,
    categories: str = "categories",
    category_id: str = "category_id",
) -> None:
    """Explore dataset in [detection](https://gradiant.github.io/ai-dataset-template/supported_tasks/#detection) format.

    Args:
        ground_truth_file:
            path to file in [detection](https://gradiant.github.io/ai-dataset-template/supported_tasks/#detection) format.
        output_folder:
            plotting results will be generated in this folder.
            If it does not exists, it will be created.
        categories:
            name of the top level key holding information about the list of categories.
        category_id:
            name of the annotation level key holding information about the category index.
    """
    import pandas as pd
    from plotly import express as px

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    ground_truth = json.load(open(ground_truth_file))

    category_id_to_name = {
        category["id"]: category["name"] for category in ground_truth[categories]
    }
    image_id_to_shape = {
        image["id"]: (image["width"], image["height"])
        for image in ground_truth["images"]
    }
    image_id_to_file_name = {
        image["id"]: image["file_name"] for image in ground_truth["images"]
    }

    images_df = pd.DataFrame(ground_truth["images"])

    images_df.to_csv(
        Path(output_folder) / f"{Path(ground_truth_file).stem}_images.csv", index=False,
    )

    px.scatter(images_df, x="width", y="height").write_html(
        f"{output_folder}/image_shape_distribution.html"
    )

    annotations_dict = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        image_width, image_height = image_id_to_shape[annotation["image_id"]]
        annotations_dict["category"].append(
            category_id_to_name[annotation[category_id]]
        )
        annotations_dict["file_name"].append(
            image_id_to_file_name[annotation["image_id"]]
        )
        annotations_dict["width"].append(annotation["bbox"][2])
        annotations_dict["height"].append(annotation["bbox"][3])
        annotations_dict["abs_width"].append(annotation["bbox"][2] / image_width)
        annotations_dict["abs_height"].append(annotation["bbox"][3] / image_height)

    annotations_df = pd.DataFrame(ground_truth["annotations"])
    for k, v in annotations_dict.items():
        annotations_df[k] = v

    annotations_df.to_csv(
        Path(output_folder) / f"{Path(ground_truth_file).stem}_annotations.csv",
        index=False,
    )

    px.histogram(annotations_df, x="category", y="area", histfunc="sum").write_html(
        f"{output_folder}/areas_per_category.html"
    )

    px.histogram(annotations_df, x="category").write_html(
        f"{output_folder}/count_per_category.html"
    )

    px.scatter(
        annotations_df,
        x="abs_width",
        y="abs_height",
        color="category",
        hover_data=["file_name", "width", "height"],
    ).write_html(f"{output_folder}/bounding_box_shape_distribution.html")


if __name__ == "__main__":
    fire.Fire(explore_ground_truth_detection)
