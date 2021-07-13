"""Module for splitting the given data and bring it in the right format for training."""

import os
import re
import math
from collections import Counter
from typing import Dict
from typing import NoReturn
from utils.io_utils import save_dict

from sklearn.model_selection import train_test_split

from data_processing.data_collector import Collector
from data_processing.abstract_classes.abstract_creator import AbstractDataSetCreator
from data_processing.data_annotator import ANNOTATION_STR

from utils.converter_utils import extract_classes

EMPTY_ANNOTATION = "empty"
UNMATCHED_STR = "<unmatched>"
CATEGORY_MAP_NAME = "categories.json"

PRINTOUTS = {
    0: "Please type in a {0} string that determines the category: ",
    1: "Please type in a regex expression that further determines "
    "the category string. If not necessary, type in '.': ",
}


class DataSetCreator(AbstractDataSetCreator):
    """
    Args:
        collector (Collector): Collector with file paths
        to split in different sets.
        annotation_folder (str): Folder with assigned
        annotations.
        out_folder (str): Folder to save splitted file
        paths to.
        data_split (Dict[str, float]): Ratios for splits.
    """

    def __init__(
        self,
        collector: Collector,
        annotation_folder: str,
        out_folder: str,
        data_split: Dict[str, float],
    ) -> None:
        AbstractDataSetCreator.__init__(
            self, collector, annotation_folder, out_folder, data_split
        )

    def create_split(self) -> NoReturn:
        """Perform the selected split strategy."""
        getattr(self, self.strategy)()

    def random_split_strategy(self) -> NoReturn:
        """Splits data and annotations randomly by given train/val/test ratio
        and preserves the class distributions.
        """
        class_list = extract_classes(self.annotations, ANNOTATION_STR, EMPTY_ANNOTATION)

        train_data, test_data, train_annotation, test_annotation = train_test_split(
            self.collector,
            class_list,
            train_size=self.train_size,
            stratify=class_list,
            random_state=42,
        )
        val_data, test_data, val_annotation, test_annotation = train_test_split(
            test_data,
            test_annotation,
            train_size=self.val_size / (self.val_size + self.test_size),
            stratify=test_annotation,
            random_state=42,
        )
        classes_per_split = [
            ("train", train_annotation),
            ("val", val_annotation),
            ("test", test_annotation),
        ]
        split_counter = {
            split_name: Counter(class_names)
            for split_name, class_names in classes_per_split
        }

        split_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

        self.move_files(split_dict)
        self.save_info_file(split_counter)

    def preserve_category_strategy(self) -> NoReturn:
        """Splits data and annotations randomly by given train/val/test ratio
        and preserves the class distributions.
        """
        class_list = extract_classes(self.annotations, ANNOTATION_STR, EMPTY_ANNOTATION)

        start_str = self.input_handler.expect_input(PRINTOUTS[0].format("start"))
        end_str = self.input_handler.expect_input(PRINTOUTS[0].format("end"))
        expr = self.input_handler.expect_input(PRINTOUTS[1])

        category_dict = {}
        for item, annotation in zip(self.collector, class_list):
            try:
                result = re.search(f"{start_str}({expr}*){end_str}", item).group(1)
            except Exception:
                result = UNMATCHED_STR
            if result in category_dict:
                category_dict[result].append((item, annotation))
            else:
                category_dict[result] = [(item, annotation)]

        adjust_unmatched = 1 if UNMATCHED_STR in category_dict else 0

        n_categories = len(category_dict) - adjust_unmatched

        n_train = math.ceil(n_categories * self.train_size)
        n_val = math.floor(n_categories * self.val_size)

        train_data, val_data, test_data = [], [], []
        train_annotation, val_annotation, test_annotation = [], [], []

        for i, key in enumerate(category_dict):
            for item in category_dict[key]:
                if i in range(n_train):
                    train_data.append(item[0])
                    train_annotation.append(item[1])
                elif i in range(n_train, n_train + n_val):
                    val_data.append(item[0])
                    val_annotation.append(item[1])
                else:
                    test_data.append(item[0])
                    test_annotation.append(item[1])

        classes_per_split = [
            ("train", train_annotation),
            ("val", val_annotation),
            ("test", test_annotation),
        ]
        split_counter = {
            split_name: Counter(class_names)
            for split_name, class_names in classes_per_split
        }

        split_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

        self.move_files(split_dict)
        self.save_info_file(split_counter)
        category_map = {}
        for category in category_dict:
            for path, _ in category_dict[category]:
                category_map[os.path.basename(path)] = category
        save_dict(category_map, os.path.join(self.out_folder, CATEGORY_MAP_NAME))
