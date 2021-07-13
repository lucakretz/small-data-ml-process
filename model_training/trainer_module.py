"""Implementation of trainer class for training of the lightning module."""

from typing import Dict
from typing import Any
from typing import NoReturn

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class TrainerModule(Trainer):  # pylint: disable=too-many-ancestors
    """Modified Training class for the definition of
    the training framework.

    Args:
        config (Dict[str, Any]): Training framework parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> NoReturn:
        super().__init__(
            logger=TensorBoardLogger(config["log_dir"], config["log_folder"])
            if config["logger"]
            else None,
            callbacks=[
                ModelCheckpoint(
                    dirpath=config["checkpoint_folder"],
                    filename=config["checkpoint_filename"],
                    monitor=config["monitor"],
                    mode=config["mode"],
                    save_top_k=1,
                )
            ],
            default_root_dir=config["root_dir"],
            gpus=config["gpus"],
            check_val_every_n_epoch=config["check_val_every_n_epoch"],
            max_epochs=config["max_epochs"],
            profiler=config["profiler"],
        )
        self.convert_to_onnx = config["convert_to_onnx"]

    def on_fit_end(self) -> NoReturn:
        """Perform onnx conversion on end of training."""
        if self.convert_to_onnx:
            for callback in self.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_path = callback.best_model_path
                    model = self.model.load_from_checkpoint(checkpoint_path).eval()
                    model.to_onnx(
                        checkpoint_path.replace("ckpt", "onnx"),
                        export_params=True,
                    )
