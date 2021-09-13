import json
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def explore_ground_truth_segmentation(
    ground_truth_file: str,
    output_folder: str,
    categories: str = "categories",
    category_id: str = "category_id",
) -> None:
    """Explore dataset in [segmentation](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation) format.

    Args:
        ground_truth_file:
            path to file in [segmentation](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation) format.
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

    images_df = pd.DataFrame(ground_truth["images"])

    images_df.to_csv(
        Path(output_folder) / f"{Path(ground_truth_file).stem}_images.csv", index=False,
    )

    px.scatter(images_df, x="width", y="height").write_html(
        f"{output_folder}/image_shape_distribution.html"
    )

    annotations_df = pd.DataFrame(ground_truth["annotations"])
    annotations_df["category"] = [
        category_id_to_name[x] for x in annotations_df["category_id"]
    ]

    annotations_df.to_csv(
        Path(output_folder) / f"{Path(ground_truth_file).stem}_annotations.csv",
        index=False,
    )

    px.histogram(annotations_df, x="category", y="area", histfunc="sum").write_html(
        f"{output_folder}/areas_per_category.html"
    )


if __name__ == "__main__":
    fire.Fire(explore_ground_truth_segmentation)
