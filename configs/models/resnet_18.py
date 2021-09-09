# type: ignore
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="ResNet", depth=18, num_stages=4, out_indices=(3,), style="pytorch"
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=512,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1, 5),
    ),
)

load_from = "/media/VA/pretrained_weights/mmcls/resnet18_batch256_20200708-34ab8f90.pth"
