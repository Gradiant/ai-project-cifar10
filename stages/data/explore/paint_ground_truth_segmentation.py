import json
from collections import defaultdict
from pathlib import Path

import fire
from loguru import logger


def annotation_to_polygons(annotation, width, height):
    import cv2
    import numpy as np
    from pycocotools import mask as cocomask

    if type(annotation["segmentation"]) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(annotation["segmentation"], height, width)
        rle = cocomask.merge(rles)
    elif type(annotation["segmentation"]["counts"]) == list:
        # uncompressed RLE
        rle = cocomask.frPyObjects(annotation["segmentation"], height, width)
    else:
        # compressed rle
        rle = annotation["segmentation"]

    mask = cocomask.decode(rle)

    major, minor, *rest = cv2.__version__.split(".")
    if int(major) < 4:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        image, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        contours, hierarchy = cv2.findContours(
            (mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

    polygons = []

    for contour in contours:
        if len(contour) >= 4:
            polygons.append(contour.squeeze())

    return polygons


@logger.catch(reraise=True)
def paint_ground_truth_segmentation(
    ground_truth_file: str,
    ground_truth_image_folder: str,
    output_folder: str,
    categories: str = "categories",
    category_id: str = "category_id",
    opacity: float = 0.5,
) -> None:
    """Paint ground truth in [segmentation](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation) format.

    Args:
        ground_truth_file:
            path to file in [segmentation](https://gradiant.github.io/ai-dataset-template/supported_tasks/#segmentation) format.
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

        width, height = image_pil.size
        fig = px.imshow(image_pil)
        fig.update_layout(title=Path(file_name).name)

        for annotation in image_id_to_annotations[image_id]:
            polygons = annotation_to_polygons(annotation, width, height)
            color = px.colors.qualitative.Light24[annotation[category_id] % 24]

            for polygon in polygons:
                fig.add_trace(
                    go.Scatter(
                        x=polygon[:, 0],
                        y=polygon[:, 1],
                        fill="toself",
                        name=category_id_to_name[annotation[category_id]],
                        fillcolor=color,
                        line_color=color,
                        opacity=opacity,
                    )
                )

        output_name = Path(output_folder) / Path(file_name).name
        logger.info(f"Saving to {output_name}")
        fig.write_html(str(output_name.with_suffix(".html")))


if __name__ == "__main__":
    fire.Fire(paint_ground_truth_segmentation)
