"""Implemented abstract Creator class used as blueprint."""

from abc import ABC, abstractmethod
import os
from typing import Dict
from typing import List
from typing import Any
from typing import NoReturn

from logging import getLogger

from data_processing.data_collector import Collector
from data_processing.helpers.input_handler import InputHandler
from data_processing.helpers.type_handler import TypeHandler

from utils.console_utils import show_data
from utils.io_utils import load_annotation_file
from utils.io_utils import copy_file
from utils.io_utils import mkdirr


STRATEGY_IDENTIFIER = "strategy"
SPLITTING_HEADER = ["ID", "Split strategy"]
SPLIT_KEYS = ["train", "val", "test"]
DATA_FOLDER = "data"
ANNOTATION_FOLDER = "annotations"
INFO_FILE_NAME = "info.txt"
PRINTOUTS = {
    0: "\nSelect a splitting strategy to be performed. "
    "Type in the selection of ID for the assigned strategy:",
}

type_handler = TypeHandler()
logger = getLogger(__file__)


class AbstractDataSetCreator(ABC):  # pylint: disable=too-many-instance-attributes
    """Module to inherit from when setting up a new
    data set creator.

    Args:
        collector (Collector): Collection of files.
        annotation_folder (str): Folder with assigned annotation
        files.
        out_folder (str): Folder to save the splitted date.
        data_split (Dict[str, float]): Dictionary that defines
        the ratios for train/val/test set.
    """

    def __init__(
        self,
        collector: Collector,
        annotation_folder: staticmethod,
        out_folder: str,
        data_split: Dict[str, float],
    ) -> None:
        super().__init__()
        self.input_handler = InputHandler()
        self.collector = collector
        self.annotation_folder = annotation_folder
        self.annotations = self._get_annotations()
        self.out_folder = out_folder
        self.data_split = data_split
        self._check_data_split()

    def _get_annotations(self) -> List[Dict[int, Dict[str, Any]]]:
        """Extracts annotations from the files.
        Structure has to follow the format perdefined in the annotator class.

        Returns:
            List[Dict[int, Dict[str, Any]]]: Annotations in dictionary format.
        """
        annotations = []
        for item in self.collector:
            data_file_type = os.path.basename(item).split(".")[-1]
            annotations.append(
                load_annotation_file(
                    os.path.join(
                        self.annotation_folder,
                        os.path.basename(item).replace(data_file_type, "json"),
                    )
                )
            )

        return annotations

    def _get_strategies(self) -> Dict[str, str]:
        """Lists all methods that have the STRATEGY_IDENTIFIER
        in their names. These methods can be selected as a part of the
        analysis pipeline.

        Returns:
            Dict[str, str]: Possible selection for the pipeline components.
        """
        strategies = [method for method in dir(self) if STRATEGY_IDENTIFIER in method]

        if not strategies:
            logger.warning(
                "There are no strategy provided. "
                "Make sure the implemented strategy methods "
                "start contain the '%s' term." % STRATEGY_IDENTIFIER
            )
        return {str(n_method): method for n_method, method in enumerate(strategies)}

    def select_strategies(self) -> NoReturn:
        """Expects user input that determins the splitting
        strategy that is performed.
        """
        strategy_map = self._get_strategies()
        show_data(strategy_map, header=SPLITTING_HEADER, key2row=True)
        strategy_id = self.input_handler.expect_input(PRINTOUTS[0])

        self.input_handler.check_input(strategy_id, strategy_map.keys())
        self.strategy = strategy_map[strategy_id]

    def _check_data_split(self):
        if sorted(SPLIT_KEYS) != sorted(list(self.data_split.keys())):
            raise KeyError("Keys in data_split are unequal to %s." % SPLIT_KEYS)
        if sum(list(self.data_split.values())) > 1:
            raise ValueError(
                "The ratios given in the data_split "
                "have to add up to 1."
            )
        self.train_size = self.data_split[SPLIT_KEYS[0]]
        self.val_size = self.data_split[SPLIT_KEYS[1]]
        self.test_size = self.data_split[SPLIT_KEYS[2]]

    def move_files(self, file_dict: Dict[str, List[str]]) -> NoReturn:
        """Move the files in the train/val/test directories according
        to the previously defined split.

        Args:
            file_dict (Dict[str, List[str]]): Dictionary that assignes
            the files to split folder.
        """

        for folder in file_dict:
            target_folder = os.path.join(self.out_folder, folder)
            mkdirr(target_folder)
            for file_path in file_dict[folder]:
                annotation_file_name = (
                    os.path.basename(file_path)
                    .replace("png", "json")
                    .replace("jpg", "json")
                )
                annotation_file_path = os.path.join(
                    self.annotation_folder, annotation_file_name
                )

                copy_file(file_path, os.path.join(target_folder, DATA_FOLDER))
                copy_file(
                    annotation_file_path, os.path.join(target_folder, ANNOTATION_FOLDER)
                )

    def save_info_file(self, info_dict: Dict[str, Dict[str, int]]) -> NoReturn:
        """Stores a txt file with the distribution of the classes across
        the split folders.

        Args:
            info_dict (Dict[str, Dict[str, int]]): Class count per split.
        """
        with open(os.path.join(self.out_folder, INFO_FILE_NAME), "w") as file_handle:
            for split in info_dict:
                file_handle.write(split + ":\n")
                for class_name in info_dict[split]:
                    file_handle.write(
                        class_name + ": " + str(info_dict[split][class_name]) + "\n"
                    )
                file_handle.write("\n")

    @abstractmethod
    def create_split(self) -> NoReturn:
        """Performes the selected split strategy.

        Raises:
            NotImplementedError: Creator has to have
            a create split method.
        """
        raise NotImplementedError
