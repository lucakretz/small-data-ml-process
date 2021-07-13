"""Class for handling and extracting file types."""

import os
from typing import List
from typing import Union

from data_processing.data_collector import Collector

TYPE_MAP = {
    "png": "png",
    "json": "json",
    "jpg": "jpg",
    "jpeg": "jpg",
    "xml": "xml",
    "txt": "txt",
}
UNKNOWN_TOKEN = "<UNKNOWN>"


class TypeHandler:
    """Module to extract file types from files and
    set the supported file endings.
    """

    @staticmethod
    def _get_file_type(file_path: str) -> str:
        """Retreive file type from file path.

        Args:
            file_path (str): Path of file.

        Returns:
            str: File type if supported.
        """
        file_name = os.path.basename(file_path)
        if file_name.count(".") == 0:
            return UNKNOWN_TOKEN
        else:
            ending = file_name.split(".")[-1]
            return TYPE_MAP.get(ending, UNKNOWN_TOKEN)

    def get_types(
        self, path_list: Union[Collector, List[str], str]
    ) -> List[str]:
        """Get file types for a whole collection.

        Args:
            path_list (Union[Collector, List[str], str]): List
            of file paths.

        Returns:
            List[str]: Detected file types.
        """
        if isinstance(path_list, str):
            path_list = [path_list]
        return list(map(self._get_file_type, path_list))
