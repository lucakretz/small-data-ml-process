"""Orchestrator for the preprocessing and training functionalities."""

import click
from logging import getLogger
from typing import Dict
from typing import Any
from typing import NoReturn

from data_processing.data_collector import Collector
from data_processing.data_reader import Reader
from data_processing.data_annotator import Annotator
from data_processing.data_creator import DataSetCreator

from model_training.transformation import get_augmentations
from model_training.data_module import DataModule
from model_training.trainer_module import TrainerModule
from model_training.model_module import EncoderModule
from model_training.model_module import TaskModule

from utils import parser

logger = getLogger(__file__)


def perform_transformation(config: Dict[str, str]) -> NoReturn:
    """The process to read new data and transform it to the required formats.

    Args:
        config (Dict[str, str]): Configurations required
        for the label transformation pipeline.
    """
    collector = Collector(config["data"]["collector_mode"])
    collector.collect(config["data"]["in_path"])
    reader = Reader(
        config["data"]["type"],
        config["data"]["out_path"],
        config["data"]["transfer_files"],
    )
    reader.select_file_type(collector)
    reader.read()
    annotator = Annotator(reader)
    annotator.produce_annotation_files(config["data"]["out_path"])


def perform_creation(config: Dict[str, str]) -> NoReturn:
    """The process to create new train/val/test set for training.

    Args:
        config (Dict[str, str]): Configurations
        required for the data set creation pipeline.
    """
    collector = Collector(config["data"]["collector_mode"])
    collector.collect(config["data"]["in_path"])
    reader = Reader(config["data"]["type"])
    reader.select_file_type(collector)
    creator = DataSetCreator(
        collector,
        config["data"]["annotation_folder"],
        config["data"]["out_path"],
        config["data"]["data_split"],
    )
    creator.select_strategies()
    creator.create_split()


def perform_training(config: Dict[str, Any]) -> NoReturn:
    """Training pipeline for training new task dependant model.

    Args:
        config (Dict[str, Any]): Config that provides pipeline parameters.
    """
    # train encoder
    encoder_config = config["encoder"]
    data_module = DataModule(
        **encoder_config["data"],
        train_transforms=get_augmentations(encoder_config["data"]["image_dim"]),
    )
    trainer = TrainerModule(encoder_config["trainer_args"])
    model = EncoderModule(**encoder_config["hparams"])
    trainer.fit(model, datamodule=data_module)

    encoder_path = trainer.checkpoint_callback.best_model_path

    # train task specific head
    classifier_config = config["task_head"]
    data_module = DataModule(
        **classifier_config["data"],
        train_transforms=get_augmentations(classifier_config["data"]["image_dim"]),
    )
    trainer = TrainerModule(classifier_config["trainer_args"])
    model = TaskModule(**classifier_config["hparams"], encoder_path=encoder_path)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


@parser.run.command("run_transformation")
@click.pass_context
def run_transformation(context: click.Context) -> NoReturn:
    """Run data transformation task.

    Args:
        context (click.Context): Content of the transformation config json file.
    """
    config = context.obj["config"]
    perform_transformation(config)


@parser.run.command("run_creation")
@click.pass_context
def run_creation(context: click.Context) -> NoReturn:
    """Run data set creation task.

    Args:
        context (click.Context): Content of the creation config json file.
    """
    config = context.obj["config"]
    perform_creation(config)


@parser.run.command("run_training")
@click.pass_context
def run_training(context: click.Context) -> NoReturn:
    """Run model training.

    Args:
        context (click.Context): Content of the training config json file.
    """
    config = context.obj["config"]
    perform_training(config)


if __name__ == "__main__":
    parser.run()  # pylint: disable=no-value-for-parameter
