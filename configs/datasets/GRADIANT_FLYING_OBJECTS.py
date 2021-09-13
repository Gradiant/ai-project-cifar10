dataset_type = "ImageNet"

img_dir = "/media/BM/databases/CIFAR10/"
ann_dir = "results/data/transform/coco_to_mmclassification-CIFAR10/"

CLASSES = ["airplane", "automobile","bird","cat","deer","dog","frog","horse","ship","truck"] 

img_scale = (32, 32)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=img_scale),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=img_scale),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=img_dir,
        ann_file=ann_dir + "CIFAR-10_training.txt",
        pipeline=train_pipeline,
        classes=CLASSES,
    ),
    val=dict(
        type=dataset_type,
        data_prefix=img_dir,
        ann_file=ann_dir + "CIFAR-10_testing.txt",
        pipeline=test_pipeline,
        classes=CLASSES,
    ),
    test=dict(
        type=dataset_type,
        data_prefix=img_dir,
        ann_file=ann_dir + "CIFAR-10_testing.txt",
        pipeline=test_pipeline,
        classes=CLASSES,
    ),
)
