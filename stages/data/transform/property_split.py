import json
import re
from pathlib import Path
from typing import Optional

import fire
from loguru import logger


@logger.catch(reraise=True)  # noqa: C901
def property_split(
    annotations_file: str,
    output_matched_file: str,
    property_name: str,
    pattern_to_match: str,
    output_unmatched_file: Optional[str] = None,
) -> None:
    """Split based on matching each image's `property_name` with `pattern_to_match`.

    Args:
        annotations_file:
            File in COCO Classification/Detection/Segmentation format.
        output_matched_file:
            File where split of matched images will be saved.
        property_name:
            name of the string property that must be present in all items of `images`
            of `annotation_file`.
        pattern_to_match:
            Regex pattern to match `property_name`
        output_unmatched_file:
            If not None, split of unmatched images will be saved to this path.

    """
    Path(output_matched_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading annotations from {annotations_file} ...")
    with open(annotations_file) as anns:
        data = json.load(anns)

    output_split = {k: v for k, v in data.items() if k not in ("images", "annotations")}

    logger.info("Processing images ...")
    matched_images = []
    unmatched_images = []
    for image in data["images"]:
        if re.match(pattern_to_match, image[property_name]):
            matched_images.append(image)
        else:
            unmatched_images.append(image)
    output_split["images"] = matched_images

    logger.info("Processing annotations ...")
    matched_image_ids = set(x["id"] for x in output_split["images"])
    matched_annotations = []
    unmatched_annotations = []
    for annotation in data["annotations"]:
        if annotation["image_id"] in matched_image_ids:
            matched_annotations.append(annotation)
        else:
            unmatched_annotations.append(annotation)

    output_split["annotations"] = matched_annotations

    logger.info(f"Writing {output_matched_file}...")
    with open(output_matched_file, "w") as f:
        json.dump(output_split, f, indent=2)
    logger.info("Done!")

    if output_unmatched_file is not None:
        output_split["images"] = unmatched_images
        output_split["annotations"] = unmatched_annotations
        logger.info(f"Writing {output_unmatched_file}...")
        with open(output_unmatched_file, "w") as f:
            json.dump(output_split, f, indent=2)
        logger.info("Done!")


if __name__ == "__main__":
    fire.Fire(property_split)
