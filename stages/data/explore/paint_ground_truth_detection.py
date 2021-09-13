import json
from collections import defaultdict
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def paint_ground_truth_detection(
    ground_truth_file: str,
    ground_truth_image_folder: str,
    output_folder: str,
    categories: str = "categories",
    category_id: str = "category_id",
    opacity: float = 0.5,
) -> None:
    """Paint ground truth in [detection](https://gradiant.github.io/ai-dataset-template/supported_tasks/#detection) format.

    Args:
        ground_truth_file:
            path to file in [detection](https://gradiant.github.io/ai-dataset-template/supported_tasks/#detection) format.
        ground_truth_image_folder:
            path to file of images associated with `ground_truth_file`.
        output_folder:
            plotting results will be generated in this folder.
            If it does not exists, it will be created.
        categories:
            name of the top level key holding information about the list of categories.
        category_id:
            name of the annotation level key holding information about the category index.
        opacity:
            Set level of opacity of the annotations painted on top of the image.
            From 0 (transparent) to 1 (opaque).
    """
    from PIL import Image
    from plotly import express as px
    from plotly import graph_objects as go

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    ground_truth = json.load(open(ground_truth_file))

    category_id_to_name = {
        category["id"]: category["name"] for category in ground_truth[categories]
    }

    image_id_to_file_name = {
        image["id"]: image["file_name"] for image in ground_truth["images"]
    }

    image_id_to_annotations = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    for image_id, annotation in image_id_to_annotations.items():
        file_name = image_id_to_file_name[image_id]

        logger.info(f"Loading {Path(ground_truth_image_folder) / file_name}")
        try:
            image_pil = Image.open(Path(ground_truth_image_folder) / file_name)
        except FileNotFoundError:
            logger.warning(f"{file_name} not found")
            continue

        fig = px.imshow(image_pil)
        fig.update_layout(title=Path(file_name).name)

        for annotation in image_id_to_annotations[image_id]:
            color = px.colors.qualitative.Light24[annotation[category_id] % 24]
            left, top, width, height = annotation["bbox"]
            right = left + width
            bottom = top + height
            fig.add_trace(
                go.Scatter(
                    x=[left, right, right, left],
                    y=[top, top, bottom, bottom],
                    fill="toself",
                    name=category_id_to_name[annotation[category_id]],
                    fillcolor=color,
                    line_color=color,
                    opacity=opacity,
                )
            )

        output_name = (Path(output_folder) / file_name).with_suffix(".html")
        logger.info(f"Saving to {output_name}")
        fig.write_html(str(output_name))


if __name__ == "__main__":
    fire.Fire(paint_ground_truth_detection)
