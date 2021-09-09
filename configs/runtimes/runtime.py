checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="MlflowLoggerHook", exp_name="cifar10"),
    ],
)
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
resume_from = None
workflow = [("train", 1)]
