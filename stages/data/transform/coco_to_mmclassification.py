import json
from typing import Optional

import fire
from loguru import logger


@logger.catch(reraise=True)
def coco_to_mmclassification(
    annotations_file: str, output_file: str, subtract_one: Optional[bool] = False,
):
    """Convert json in [classification format](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification) to txt in [mmclassification format](https://mmclassification.readthedocs.io/en/latest/tutorials/new_dataset.html#reorganize-dataset-to-existing-format).

    Args:
        annotations_file:
            path to json in [classification format](https://gradiant.github.io/ai-dataset-template/supported_tasks/#classification)
        output_file:
            path to write the txt in [mmclassification format](https://mmclassification.readthedocs.io/en/latest/tutorials/new_dataset.html#reorganize-dataset-to-existing-format)
        subtract_one:
            Default False.
            mmclassification format expect the class indices to start at 0;
            if your `category_id` starts at 1, set this to True
            in order to subtract one to the values.
    """
    logger.info("Loading annotations")
    annotations = json.load(open(annotations_file))

    category_name_to_id = {
        category["name"]: category["id"] for category in annotations["categories"]
    }

    if min(category_name_to_id.values()) != 0 and not subtract_one:
        raise ValueError(
            "mmclassification format expect the class indices to start at 0; "
            "if your `category_id` in COCO start at 1, set this to True "
            "in order to subtract one to the values."
        )

    with open(output_file, "w") as f:
        for image in annotations["images"]:
            if subtract_one:
                image["category_id"] -= 1
            line = f"{image['file_name']} {image['category_id']}"
            f.write(line)
            f.write("\n")

    return output_file


if __name__ == "__main__":
    fire.Fire(coco_to_mmclassification)
