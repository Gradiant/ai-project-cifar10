import json
import random
from pathlib import Path
from typing import Optional

import fire
from loguru import logger


@logger.catch(reraise=True)  # noqa: C901
def random_split_classification(
    annotations_file: str,
    output_matched_file: str,
    val_proportion: float,
    random_seed: Optional[int] = 47,
    output_unmatched_file: Optional[str] = None,
) -> None:
    """Split images and annotations randomly into `matched` and `unmatched` sets.

    Args:
        annotations_file:
            File in COCO [classification](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification) format.
        output_matched_file:
            File where split of matched images will be saved.
        val_proportion:
            Proportion of validation images. Expected values between 0.0 and 1.0.
        random_seed:
            Seed for reproducibility.
        output_unmatched_file:
            If not None, split of unmatched images will be saved to this path.

    """
    Path(output_matched_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading annotations from {annotations_file} ...")
    with open(annotations_file) as anns:
        data = json.load(anns)

    output_split = {k: v for k, v in data.items() if k not in ("images")}

    logger.info("Processing images ...")
    matched_images = []
    unmatched_images = []
    random.seed(random_seed)
    for image in data["images"]:
        if random.random() < val_proportion:
            matched_images.append(image)
        else:
            unmatched_images.append(image)
    output_split["images"] = matched_images

    logger.info(f"Writing {output_matched_file}...")
    with open(output_matched_file, "w") as f:
        json.dump(output_split, f, indent=2)
    logger.info("Done!")

    if output_unmatched_file is not None:
        output_split["images"] = unmatched_images
        logger.info(f"Writing {output_unmatched_file}...")
        with open(output_unmatched_file, "w") as f:
            json.dump(output_split, f, indent=2)
        logger.info("Done!")


if __name__ == "__main__":
    fire.Fire(random_split_classification)
