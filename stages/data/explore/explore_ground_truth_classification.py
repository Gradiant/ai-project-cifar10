import json
from collections import defaultdict
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def explore_ground_truth_classification(
    ground_truth_file: str,
    output_folder: str,
    categories: str = "categories",
    category_id: str = "category_id",
) -> None:
    """Explore dataset in [classification](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification) format.

    Args:
        ground_truth_file:
            path to file in [classification](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification) format.
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

    ground_truth_info = defaultdict(list)
    for image in ground_truth["images"]:
        ground_truth_info["category"].append(category_id_to_name[image[category_id]])
        ground_truth_info["width"].append(image["width"])
        ground_truth_info["height"].append(image["height"])

    ground_truth_df = pd.DataFrame(ground_truth_info)

    ground_truth_df.to_csv(
        Path(output_folder) / Path(ground_truth_file).with_suffix(".csv").name,
        index=False,
    )
    px.bar(ground_truth_df["category"].value_counts()).write_html(
        f"{output_folder}/category_distribution.html"
    )
    px.scatter(ground_truth_df, x="width", y="height", color="category").write_html(
        f"{output_folder}/image_shape_distribution.html"
    )


if __name__ == "__main__":
    fire.Fire(explore_ground_truth_classification)
