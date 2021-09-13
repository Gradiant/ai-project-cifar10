# optimizer
optimizer = dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="cyclic", target_ratio=(10, 1e-2), step_ratio_up=0.3)

total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=1, metric="accuracy")
workflow = [("train", 1)]