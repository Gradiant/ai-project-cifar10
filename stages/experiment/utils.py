def get_task_lib(task):
    task_to_lib = {
        "detection": "mmdet",
        "classification": "mmcls",
        "segmentation": "mmseg",
    }
    return task_to_lib[task]


def get_model_type(task):
    model_type = {
        "detection": "detector",
        "classification": "model",
        "segmentation": "segmentor",
    }
    return model_type[task]
