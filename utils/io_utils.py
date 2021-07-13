"""General Utilities for data loading and saving."""

import os
import json
from shutil import copy
from typing import Any
from typing import Dict
from typing import Union
from typing import NoReturn
from logging import getLogger

logger = getLogger(__file__)


def load_annotation_file(file_path: str) -> Dict[int, Dict[str, Any]]:
    """Load dictionary object from annotation file.

    Args:
        file_path (str): Path pointing to file.

    Raises:
        FileNotFoundError: Raises if file doesn't exist.

    Returns:
        Dict[int, Dict[str, Any]]: All annotations stored in a dict
        assigned to one data sample.
    """
    if os.path.isfile(file_path) and file_path.endswith(".json"):
        with open(file_path, "r") as file_handle:
            return json.load(file_handle)
    else:
        raise FileNotFoundError("'%s' is not an valid annotation file." % file_path)


def save_dict(dictionary: Dict[Union[int, str], Any], file_path: str) -> NoReturn:
    """Save a dictionary as json.

    Args:
        dictionary (Dict[Any]): Dict with data.
        file_path (str): Path for json file.
    """
    with open(file_path, "w") as file_handle:
        json.dump(dictionary, file_handle)


def copy_file(src_file: str, target_folder: str) -> NoReturn:
    """Copies file into other directory.

    Args:
        src_file (str): File to copy.
        target_folder (str): Folder to store copy of file.

    Raises:
        FileNotFoundError: Raises if copy subject doesn't exist.
    """
    mkdirr(
        os.path.dirname(target_folder)
        if target_folder.endswith((".png", ".jpg"))
        else target_folder
    )
    try:
        copy(src=src_file, dst=target_folder)
    except FileNotFoundError as error:
        raise FileNotFoundError("'%s' does not exist." % src_file) from error


def mkdirr(directory: str) -> NoReturn:
    """Creates a directory recursively.

    Args:
        directory (str): Directory to create.

    Raises:
        NotADirectoryError: Raises if directory couldn't be created.
    """

    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except Exception as error:
            raise NotADirectoryError(
                "'%s' could not be created!" % directory
            ) from error
