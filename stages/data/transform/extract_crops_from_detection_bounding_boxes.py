import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import fire
from loguru import logger


@logger.catch(reraise=True)
def extract_crops_from_detection_bounding_boxes(
    annotations_file: str,
    image_folder: str,
    output_folder: str,
    output_file: Optional[str] = None,
    categories: str = "categories",
    category_id: str = "category_id",
    expand_to_min_height: Optional[int] = None,
    expand_to_min_width: Optional[int] = None,
) -> None:
    """Extract crops from each annotation's `bbox` and it's corresponding image.

    Args:
        annotations_file:
            Path to file in [detection](https://gradiant.github.io/ai-dataset-template/supported_tasks/#detection) format.
        image_folder:
            Path to root folder containing the images as referenced in each image['file_name'] of `annotations_file`.
        output_folder:
            Path to folder where the crops will be saved.
            A subfolder for each annotation[category_id] will be created.
            Each crop will be saved in it's corresponding subfolder.
        output_file:
            If not None, a new annotations file in [classification format](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification)
            will be created in this path.
        categories:
            Name of the top-level key in `annotations_file` where category
            list is defined.
        category_id:
            Name of the per-annotation key in `annotations_file` where
            it's category is defined.
        expand_to_min_height:
            If not None, value in pixels.
            If any bbox's height is bellow `expand_to_min_height`,
            the bounding box will be expanded to have this height
            before extracting the crop.
            If the expanded box lies outside the image, it will
            be padded with 0s.
        expand_to_min_width:
            If not None, value in pixels.
            If any bbox's width is bellow `expand_to_min_height`,
            the bounding box will be expanded to have this width
            before extracting the crop.
            If the expanded box lies outside the image, it will
            be padded with 0s.
    """
    from PIL import Image

    annotations = json.load(open(annotations_file))

    image_id_to_name = {
        image["id"]: image["file_name"] for image in annotations["images"]
    }

    image_id_to_annotations = defaultdict(list)
    for annotation in annotations["annotations"]:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    category_id_to_name = {
        category["id"]: category["name"] for category in annotations[categories]
    }

    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True, parents=True)

    for category in category_id_to_name.values():
        output_category_folder_path = output_folder_path / Path(category)
        output_category_folder_path.mkdir(exist_ok=True, parents=True)

    output_classification_images: List[Dict[str, Any]] = []
    for image in annotations["images"]:
        image_name = image_id_to_name[image["id"]]
        logger.info(f"Processing image: {image_name}")

        try:
            image_pil = Image.open(Path(image_folder) / image_name)
        except FileNotFoundError:
            logger.warning(f"{image_name} not found")
            continue

        image_annotations = image_id_to_annotations[image["id"]]

        for annotation in image_annotations:
            bbox = annotation["bbox"]
            category = category_id_to_name[annotation[category_id]]

            output_category_folder_path = output_folder_path / Path(category)

            left = bbox[0]
            top = bbox[1]
            width = bbox[2]
            height = bbox[3]
            right = left + width
            bottom = top + height

            if expand_to_min_width and width < expand_to_min_width:
                pad = expand_to_min_width - width
                left -= pad // 2
                if pad % 2:
                    left -= 1
                right += pad // 2

            if expand_to_min_height and height < expand_to_min_height:
                pad = expand_to_min_height - height
                top -= pad // 2
                if pad % 2:
                    top -= 1
                bottom += pad // 2

            crop = image_pil.crop((left, top, right, bottom))
            crop_file_name = (
                output_category_folder_path
                / f"{Path(image_name).stem}_{left}_{top}_{right}_{bottom}"
                f"{Path(image_name).suffix}"
            )
            logger.debug(f"Saving crop to {crop_file_name}")
            crop.save(crop_file_name)
            output_classification_images.append(
                {
                    "id": len(output_classification_images),
                    "file_name": f"{Path(crop_file_name).parent.name}/{Path(crop_file_name).name}",
                    "width": crop.size[0],
                    "height": crop.size[1],
                    "category_id": annotation[category_id],
                }
            )

    if output_file is not None:
        del annotations["annotations"]
        annotations["images"] = output_classification_images
        logger.info(f"Writing classification annotations to {output_file}")
        with open(output_file, "w") as f:
            json.dump(annotations, f, indent=2)


if __name__ == "__main__":
    fire.Fire(extract_crops_from_detection_bounding_boxes)
