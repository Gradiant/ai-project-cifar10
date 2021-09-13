import json
from pathlib import Path

import fire
from loguru import logger


@logger.catch(reraise=True)
def process_image_segmentation(
    config_file: str,
    model_checkpoint: str,
    image_path: str,
    output_folder: str,
    device: str = "cuda:0",
) -> None:
    """Process image with [segmentation](https://gradiant.github.io/ai-project-template/supported_tasks/#segmentation) model generating image result and COCO annotations file.

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
    from mmseg.apis import inference_segmentor, init_segmentor
    from PIL import Image
    from pycocotools.mask import area, encode

    model = init_segmentor(config_file, model_checkpoint, device=device)
    img = np.array(Image.open(image_path))
    width, height = img.shape[:2]
    result = inference_segmentor(model, img)[0]
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    annotations = []
    images = [dict(id=1, width=width, height=height, file_name=Path(image_path).name)]
    categories = [
        dict(supercategory="object", id=i, name=category)
        for i, category in enumerate(model.CLASSES)
    ]

    for cat_idx in range(len(model.CLASSES)):
        bin_mask = (result == cat_idx).astype(np.uint8)
        rle_mask = encode(np.asfortranarray(bin_mask))
        rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
        annotations.append(
            dict(
                id=cat_idx,
                segmentation=rle_mask,
                area=int(area(rle_mask)),
                image_id=1,
                category_id=categories[cat_idx]["id"],
            )
        )

    coco_result = dict(categories=categories, annotations=annotations, images=images)

    with open(Path(output_folder) / "coco_results.json", "w") as f:
        json.dump(coco_result, f)


if __name__ == "__main__":
    fire.Fire(process_image_segmentation)
