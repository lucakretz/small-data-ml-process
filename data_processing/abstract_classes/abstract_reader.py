"""Implemented abstract Reader class used as blueprint."""

from abc import ABC
from abc import abstractmethod

from collections import Counter
from typing import List
from typing import Tuple
from typing import Any
from typing import NoReturn

from logging import getLogger

from data_processing.helpers.input_handler import InputHandler
from data_processing.data_collector import Collector
from data_processing.helpers.type_handler import TypeHandler
from data_processing.helpers.type_handler import UNKNOWN_TOKEN

from utils.console_utils import show_data


FILE_LISTING_HEADER = ["File formats", "Count"]
FILE_FORMAT_SEPARATOR = ";"

PRINTOUTS = {
    0: "\nThe following file types are contained by the data collection:\n",
    1: "\nPlease select the file formats from the '{0}' column\n"
    "(separate by '{1}' if multiple file formats are selected): ",
}

type_handler = TypeHandler()
logger = getLogger(__file__)


class AbstractReader(ABC):
    """Abstract class for the Reader module.
    By inheriting from this class three functionalities are available by default:
    - listing of all supported file types in file collection.
    - method to select file type(s) for further processing via command line.
    - updating the file collection by removing all not-selected file types.
    Instanciates all necessary class variables for further use.

    Args:
        ABC (class): AbstractBaseClass object.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_handler = InputHandler()
        self.information_extractors = []
        self.modes = []
        self.data = []

    @abstractmethod
    def __len__(self) -> int:
        """Defines the length of the object.

        Raises:
            NotImplementedError: Makes reader iterable.

        Returns:
            int: Number of read data.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[str, Any]:
        """Returns the path and extracted annotation/information.

        Args:
            index (int): Location of needed information.

        Raises:
            NotImplementedError: Makes Reader
            subscriptable.

        Returns:
            Tuple[str, Any]: Extracted path and
            the information of the file.
        """
        raise NotImplementedError

    def select_file_type(self, data_collection: Collector) -> NoReturn:
        """Lists the different file types contained by the collector object
        and the files that should be taken into account for the transformation
        are selected via command line.

        Args:
            data_collection (Collector): Iterative object encompassing all files found
            in directory.
        """
        # set collector for class
        self.data_collection = data_collection
        file_type_count = self.get_file_type_count(data_collection)
        print(PRINTOUTS[0])
        show_data(file_type_count, FILE_LISTING_HEADER, key2row=True)

        self.file_formats = self.input_handler.expect_input(
            PRINTOUTS[1].format(FILE_LISTING_HEADER[0], FILE_FORMAT_SEPARATOR),
            choices=[key for key in file_type_count.keys() if key != UNKNOWN_TOKEN],
            confirm=True,
        ).split(FILE_FORMAT_SEPARATOR)
        # check if input is correct
        self.input_handler.check_input(self.file_formats, file_type_count.keys())
        logger.debug("Selected formats: %s", self.file_formats)

        # only keep selected file formats
        # in collection
        self._update_collector_content(self.file_formats)

    @staticmethod
    def get_file_type_count(data_collection: Collector) -> Counter:
        """Counts the file endings found in tree.

        Args:
            data_collection (Collector): All files that
            are considered.

        Returns:
            Counter: File type counter.
        """
        counter = Counter(type_handler.get_types(data_collection))
        # replace the unrecognized filetypes
        # by UNKNOWN_TOKEN string
        if None in counter.keys():
            counter[UNKNOWN_TOKEN] = counter.pop(None)
        return counter

    def _update_collector_content(self, file_endings: List[str]) -> Collector:
        """Excludes files from collector that do not have the selected file format.

        Args:
            file_endings (List[str]): File formats that should be keept.
        """
        self.data_collection.data_files = [
            relevant_file
            for relevant_file in self.data_collection
            if any(relevant_file.endswith(ending) for ending in file_endings)
        ]
