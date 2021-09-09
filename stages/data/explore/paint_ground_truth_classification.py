import json
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def paint_ground_truth_classification(
    ground_truth_file: str,
    ground_truth_image_folder: str,
    output_folder: str,
    categories: str = "categories",
    category_id: str = "category_id",
) -> None:
    """Paint ground truth in [classification](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification) format.

    Args:
        ground_truth_file:
            path to file in [classification](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification) format.
        ground_truth_image_folder:
            path to file of images associated with `ground_truth_file`.
        output_folder:
            plotting results will be generated in this folder.
            If it does not exists, it will be created.
        categories:
            name of the top level key holding information about the list of categories.
        category_id:
            name of the annotation level key holding information about the category index.
    """
    from PIL import Image
    from plotly import express as px

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    ground_truth = json.load(open(ground_truth_file))

    category_id_to_name = {
        category["id"]: category["name"] for category in ground_truth[categories]
    }

    for image in ground_truth["images"]:
        file_name = image["file_name"]

        logger.info(f"Loading {Path(ground_truth_image_folder) / file_name}")
        try:
            image_pil = Image.open(Path(ground_truth_image_folder) / file_name)
        except FileNotFoundError:
            logger.warning(f"{file_name} not found")
            continue

        category = category_id_to_name[image[category_id]]
        fig = px.imshow(image_pil)
        fig.update_layout(title=f"{Path(file_name).name} : {category}")

        output_name = Path(output_folder) / Path(file_name).name
        logger.info(f"Saving to {output_name}")
        fig.write_html(str(output_name.with_suffix(".html")))


if __name__ == "__main__":
    fire.Fire(paint_ground_truth_classification)
