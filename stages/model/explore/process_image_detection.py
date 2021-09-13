import json
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def process_image_detection(
    config_file: str,
    model_checkpoint: str,
    image_path: str,
    output_folder: str,
    device: str = "cuda:0",
) -> None:
    """Process image with [detection](https://gradiant.github.io/ai-project-template/supported_tasks/#detection) model generating image result and COCO annotations file.

    Args:
        config_file:
            path to mmdet model config file
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
    from mmdet.apis import inference_detector, init_detector
    from PIL import Image

    model = init_detector(config_file, model_checkpoint, device=device)
    img = np.array(Image.open(image_path))
    width, height = img.shape[:2]
    results = inference_detector(model, img)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    annotations = []
    images = dict(id=1, width=width, height=height, file_name=Path(image_path).name)
    categories = [
        dict(supercategory="object", id=i, name=category)
        for i, category in enumerate(model.CLASSES)
    ]

    ann_id = 0
    for cat_idx, class_results in enumerate(results):
        for result in class_results:
            result = result.tolist()
            bbox = [result[0], result[1], result[2] - result[0], result[3] - result[1]]
            area = bbox[2] * bbox[3]
            segmentation = [
                bbox[0],
                bbox[1],
                bbox[0],
                bbox[3],
                bbox[2],
                bbox[3],
                bbox[2],
                bbox[1],
            ]
            category_id = categories[cat_idx]["id"]
            annotations.append(
                dict(
                    id=ann_id,
                    bbox=bbox,
                    segmentation=segmentation,
                    area=area,
                    image_id=1,
                    category_id=category_id,
                )
            )
            ann_id += 1

    coco_result = dict(categories=categories, annotations=annotations, images=images)

    with open(Path(output_folder) / "coco_results.json", "w") as f:
        json.dump(coco_result, f)


if __name__ == "__main__":
    fire.Fire(process_image_detection)
