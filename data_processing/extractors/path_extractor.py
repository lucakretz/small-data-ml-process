"""Module to define label by component of file paths."""

import os
from typing import List
from typing import Dict
from typing import Any
from typing import Tuple
from typing import Union
from typing import NoReturn

from data_processing.abstract_classes.abstract_extractor import (
    AbstractInformationExtractor,
)

from utils.io_utils import copy_file
from utils.converter_utils import convert_type


PRINTOUTS = {
    0: "Please provide a label-map in the format "
    "'<sub_string>:<assigned_class_number>; ...', i.e. 'red:0; blue:1': ",
    1: "The input '{0}' was not in the correct format. Please try again: ",
}

MAPPING_ID = "mapping"
FILENAME_ID = "filename"
NO_SUBSET_ID = "no subset"
TRANSFER_IMAGE_FOLDER = "images"


class PathExtractor(AbstractInformationExtractor):
    """Implemented class to assign a label based on data
    file path properties.

    Args:
        AbstractInformationExtractor (class): Blueprint for the
        implementation of the extractor.
    """

    def __init__(self) -> NoReturn:
        AbstractInformationExtractor.__init__(self)

    def extract(self) -> List[Tuple[str, List[Tuple[str, int]]]]:
        """Extract the content of the given paths:
        - Choose an attribute from the path.
        - Use this attribute to assign a label to file.

        Returns:
            List[Tuple[str, List[Tuple[str, int]]]]: Paths of the annotation
            files and assigned label structure.
        """
        paths, instances, _ = zip(*self.process_information())
        console_dict = self.show_attributes(instances)
        subset_id = self.select_attribute(
            list(console_dict.keys()),
            printout_nr=5,
            extended_choice=[NO_SUBSET_ID],
            dynamic_input=NO_SUBSET_ID,
        )
        if subset_id != NO_SUBSET_ID:
            self.subset_str = self.select_attribute(
                list(console_dict[subset_id]), printout_nr=6
            )
            paths, instances = zip(
                *[
                    (path, instance)
                    for path, instance in zip(paths, instances)
                    if instance[subset_id] == self.subset_str
                ]
            )

        attribute = self.select_attribute(
            list(console_dict.keys()),
            printout_nr=1,
            extended_choice=[MAPPING_ID],
            dynamic_input=MAPPING_ID,
        )
        if attribute == MAPPING_ID:
            label_map = self._parse_label_map()
            layer_names = self._assign_layer(
                list(console_dict.keys()), paths, instances
            )
            labels = self._assign_labels(layer_names, label_map)
        else:
            label_map = self._create_label_map(console_dict[attribute])
            labels = self._assign_labels(
                [instance[attribute] for instance in instances],
                label_map,
            )

        return list(zip(paths, labels))

    def transfer_files(
        self, old_paths: List[str], labels: List[str], out_dir: str
    ) -> List[Tuple[str, str]]:
        """Copy files to new direction with new name.

        Args:
            old_paths (List[str]): Original file paths.
            labels (List[str]): Label names.
            out_dir (str): Directory to transfer files to.

        Returns:
            List[Tuple[str, str]]: New file paths and assigned labels.
        """
        new_paths = []
        out_dir = os.path.join(out_dir, TRANSFER_IMAGE_FOLDER)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        for i, (file_name, label) in enumerate(zip(old_paths, labels)):
            new_path = os.path.join(
                out_dir, self.subset_str, label[0][0] + "_" + str(i) + ".png"
            )
            copy_file(file_name, new_path)
            new_paths.append(new_path)
        return list(zip(new_paths, labels))

    def process_information(self) -> List[Tuple[str, Dict[str, Any], None]]:
        """Get needed information for the label assignment.
        - file path.
        - dict with id assigned to path instance.

        Returns:
            List[Tuple[str, Dict[str, Any]], None]:
            Mapping of attribute for each file to an id.
        """
        information = []
        for entry in self.collector:
            components = self._split_path(entry)[1:]
            information.append(
                (entry, dict(zip(list(range(len(components))), components)), None)
            )
        return information

    def _parse_label_map(self) -> Dict[str, int]:
        """If the user decides to assign labels based on the filenames
        the needed mapping is parsed here.

        Returns:
            Dict[str, int]: Mapping that assigns substring to a class id
            provided by the user.
        """
        input_map = self.input_handler.expect_input(PRINTOUTS[0], timeout=60)
        label_map = None
        while not label_map:
            try:
                label_map = {
                    str(item.split(":")[0]).strip(): int(
                        str(item.split(":")[1]).strip()
                    )
                    for item in input_map.split(";")
                }
            except AttributeError:
                input_map = self.input_handler.expect_input(
                    PRINTOUTS[1].format(label_map)
                )
        return label_map

    def _assign_layer(
        self, layers: List[int], paths: List[str], instances: Dict[int, str]
    ) -> List[str]:
        """Assign layer to files.

        Args:
            layers (List[int]): Number of layers in path.
            paths (List[str]): File paths.
            instances (Dict[int, str]): Layer names.

        Returns:
            List[str]: [description]
        """
        layer_nr = self.select_attribute(layers, 7, extended_choice=[FILENAME_ID])

        if layer_nr == FILENAME_ID:
            names = [os.path.basename(file) for file in paths]
        else:
            layer_nr = convert_type(layer_nr, int)
            names = [instance[layer_nr] for instance in instances]

        return names

    def _assign_labels(
        self, criterion_list: List[str], mapping: Dict[str, int]
    ) -> List[int]:
        """Assigns labels to files based on attribute or file mapping.

        Args:
            criterion_list (List[str]): List of attributes to differentiate between.
            mapping (Dict[str, int]): Provided or created mapping.

        Returns:
            List[int]: List of labels in same order as files.
        """
        return [
            self._match_substring(criterion, mapping) for criterion in criterion_list
        ]

    @staticmethod
    def _match_substring(
        parent_string: str, mapping: Dict[str, int]
    ) -> Union[List[Tuple[str, int]], List[None]]:
        """If key matches a sub-string of given string
        the value is forwarded.

        Args:
            parent_string (str): Search string.
            mapping (Dict[str, int]): Mapping of substrings to id.

        Returns:
            Union[List[Tuple[str, int]], List[None]]:
            If matches the mapping value is provided
            otherwise the list is empty.
        """
        for substring in mapping:
            if substring in parent_string:
                return [(substring, mapping[substring])]
            if len(mapping) == 1:
                return [("not_" + substring, max(mapping.values()) + 1)]
        return []

    @staticmethod
    def _create_label_map(attributes: str) -> Dict[str, int]:
        """Creates dict with key to id.

        Args:
            attributes (str): List of attributes that can
            be assigned to id.

        Returns:
            Dict[str, int]: Produced mapping based on count.
        """
        return {value: label for label, value in enumerate(attributes)}

    @staticmethod
    def _split_path(path: str) -> List[str]:
        """Splits path in list of folder names.

        Args:
            path (str): File or directory path.

        Returns:
            List[str]: List of folders in path.
        """
        return os.path.dirname(path).split(os.sep)
