"""Class to collect data from every input structure."""

import os
from typing import List
from typing import NoReturn
from logging import getLogger

# read from:
# - tree: files of all subfolders
#         starting from parent folder
# - folder: all files in given folder
# - file: single file
MODES = ["tree", "folder", "file"]

logger = getLogger(__file__)


class Collector:
    """Container class for all files to proceed with
    for the annotation transformation.

    Args:
        collector_mode (str): Choose strategy for collecting files from
        'tree', 'folder', 'file'.

    Raises:
        KeyError: No valid collection strategy given.
    """

    def __init__(self, collector_mode: str) -> NoReturn:
        if collector_mode not in MODES:
            raise KeyError(
                "Selected mode '%s' is not supported. Please select from: %s"
                % (str(collector_mode), str(MODES)),
            )
        self.mode = collector_mode
        self.data_files = []

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, index: int) -> str:
        return self.data_files[index]

    @staticmethod
    def check_path(input_path: str, mode: str) -> NoReturn:
        """Checks if given paths are valid.

        Args:
            input_path (str): File or folder path.
            mode (str): Collection strategy.

        Raises:
            ValueError: Given starting folder is not existent.
            ValueError: Given file is not existent.
        """
        if not os.path.isdir(input_path) and mode in ["tree", "folder"]:
            raise ValueError(
                "For '%s' mode directory '%s' does not exist." % (mode, input_path)
            )
        if not os.path.isfile(input_path) and mode == "file":
            raise ValueError(
                "For '%s' mode directory '%s' does not exist." % (mode, input_path)
            )

    def collect(self, path: str) -> NoReturn:
        """Collect all files from tree/folder/file following given
        collecting strategy.

        Args:
            path (str): Starting path for collection strategy
        """
        self.check_path(path, self.mode)
        if self.mode == "tree":
            self.data_files = self.collect_tree(path)
        elif self.mode == "folder":
            self.data_files = self.collect_folder(path)
        else:
            self.data_files = [path]
        logger.info("Collected %i files.", len(self.data_files))

    @staticmethod
    def collect_tree(parent_folder: str) -> List[str]:
        """Go through all subfolders and store files.

        Args:
            parent_folder (str): Starting directory.

        Returns:
            List[str]: Found files.
        """
        files = []
        for dir_name, _, file_names in os.walk(parent_folder):
            files += [os.path.join(dir_name, file_name) for file_name in file_names]
        return files

    @staticmethod
    def collect_folder(folder: str) -> List[str]:
        """Store all files in given folder.

        Args:
            folder (str): Folder to search.

        Returns:
            List[str]: Found files.
        """
        files = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
        return files
