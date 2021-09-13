from pathlib import Path

import mmcv
from mmcv.image import imwrite, tensor2imgs
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class DataVisualizer(object):
    """Visualize images and annotations.

    This transform visualizes images and bboxes when used it train pipeline and just images when used it val or test pipelines.
    """

    def __init__(self, output_folder):
        self.output_folder = output_folder
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def __call__(self, results):
        img = results["img"]
        filename = Path(results["filename"]).name

        if "img_norm_cfg" in results:
            if img.ndim == 3:
                img = img[None, ...]
            img = tensor2imgs(img, **results["img_norm_cfg"])[0]

        if "gt_bboxes" in results:
            bboxes = results["gt_bboxes"]
            labels = results["gt_labels"]

            mmcv.imshow_det_bboxes(
                img,
                bboxes,
                labels,
                show=False,
                out_file=f"{self.output_folder}/{filename}",
            )
        else:
            imwrite(img, f"{self.output_folder}/{filename}")

        return results
