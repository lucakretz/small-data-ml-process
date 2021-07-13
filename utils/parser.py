"""Parser functionalities."""

import json
import click
from typing import Dict
from typing import NoReturn
from typing import Any

def validate_config(ctx: click.Context, param: click.Option, value: click.File) -> Dict[str, Any]:
    """Validate if config is a proper json and validation list items are present in config json.

    Args:
        ctx(click.Context): Click context used for passing value.
        param(click.Option): Parameter name to be validated.
        value(click.File): Config parameter value as file.
    Returns:
        Dict[str, Any]: Parsed configuration as dictionary.
    Raises:
        BadParameterException: Raised if item is missing in configurations.
    """
    if value is not None:
        configurations = json.load(value)
        return configurations
    return dict()


@click.group()
@click.option(
    "--config",
    "-c",
    callback=validate_config,
    help="Path to config file.",
    type=click.File("r"),
)
@click.pass_context
def run(ctx: click.Context, config: click.File) -> NoReturn:
    """Run function to load configuration and entry point to add additional items to the click
    command group.
    Args:
        ctx(click.Context): Click context which holds values for downstream commands.
        config(click.File): Click file which holds the configuration in json format.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = config


@click.group(chain=True)
@click.option(
    "--config",
    "-c",
    callback=validate_config,
    help="Path to config file.",
    type=click.File("r"),
)
@click.pass_context
def run_chain(ctx: click.Context, config: click.File) -> NoReturn:
    """Function to load configuration and entry point while chaining commands in sequence and
    creating a pipeline.
    Args:
        ctx(click.Context): Click context which holds values for downstream commands.
        config(click.File): Click file which holds the configuration in json format.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
