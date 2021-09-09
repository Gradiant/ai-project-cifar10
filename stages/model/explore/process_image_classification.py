import json
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def process_image_classification(
    config_file: str,
    model_checkpoint: str,
    image_path: str,
    output_folder: str,
    device: str = "cuda:0",
) -> None:
    """Process image with [classification](https://gradiant.github.io/ai-project-template/supported_tasks/#classification) model generating image result and COCO annotations file.

    Args:
        config_file:
            path to mmcls model config file
        model_checkpoint:
            pytorch `.pth` checkpoint file
        image_path:
            path to input image
        output_folder:
            plotting and annotation results will be generated in this folder.
            If it does not exists, it will be created.
        device:
            device used for inference
    """
    import numpy as np
    from mmcls.apis import inference_model, init_model
    from PIL import Image

    model = init_model(config_file, model_checkpoint, device=device)
    img = np.array(Image.open(image_path))
    width, height = img.shape[:2]
    results = inference_model(model, img)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    images = [
        dict(
            id=1,
            width=width,
            height=height,
            file_name=Path(image_path).name,
            category_id=results["pred_label"],
        )
    ]
    categories = [
        dict(supercategory="object", id=i, name=category)
        for i, category in enumerate(model.CLASSES)
    ]

    coco_result = dict(categories=categories, images=images)

    with open(Path(output_folder) / "coco_results.json", "w") as f:
        json.dump(coco_result, f)


if __name__ == "__main__":
    fire.Fire(process_image_classification)
