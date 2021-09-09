from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from fire import Fire
from loguru import logger


@logger.catch
def convert_to_torchserve(
    config_file: str,
    checkpoint_file: str,
    output_folder: str,
    model_name: Optional[str] = None,
    model_version: str = "1.0",
    force: bool = False,
):
    """Converts Open MMLab model (config + checkpoint) to TorchServe `.mar`.

    Args:
        config_file:
            In Open MMLab config format.
            The contents vary for each task repository.
        checkpoint_file:
            In Open MMLab checkpoint format.
            The contents vary for each task repository.
        output_folder:
            Folder where `{model_name}.mar` will be created.
            The file created will be in TorchServe archive format.
        model_name:
            If not None, used for naming the `{model_name}.mar` file
            that will be created under `output_folder`.
            If None, `{Path(checkpoint_file).stem}` will be used.
        model_version:
            Model's version.
        force:
            If True, if there is an existing `{model_name}.mar`
            file under `output_folder` it will be overwritten.
    """
    import mmcv
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils

    mmcv.mkdir_or_exist(output_folder)

    config = mmcv.Config.fromfile(config_file)

    if model_name is None:
        model_name = Path(checkpoint_file).stem

    with TemporaryDirectory() as tmpdir:
        config.dump(f"{tmpdir}/config.py")

        args = Namespace(
            **{
                "model_file": f"{tmpdir}/config.py",
                "serialized_file": checkpoint_file,
                "handler": f"{Path(__file__).parent}/classification_handler.py",
                "model_name": model_name or Path(checkpoint_file).stem,
                "version": model_version,
                "export_path": output_folder,
                "force": force,
                "requirements_file": None,
                "extra_files": None,
                "runtime": "python",
                "archive_format": "default",
            }
        )
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)


if __name__ == "__main__":
    Fire(convert_to_torchserve)
