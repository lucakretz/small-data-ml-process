"""Implemented abstract Extractor class used as blueprint."""

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Any
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import NoReturn

from logging import getLogger

from data_processing.data_collector import Collector
from data_processing.helpers.input_handler import InputHandler
from data_processing.helpers.type_handler import TypeHandler

from utils.console_utils import show_data
from utils.converter_utils import convert_type


type_handler = TypeHandler()
logger = getLogger(__file__)

PRINTOUTS = {
    0: "\nPossible attributes to assigne a label on:\n",
    1: "\nChoose a layer as a basis for annotation assignement "
    "or type '{0}' to provide a label map: ",
    2: "\nSelect the attribute that defines the box around the object:",
    3: "\nSelect the attribute, that defines the name of the object class:",
    4: "\nSelect the instance that contains information about object:", 
    5: "\nSelect category in which a subset identifier is contained.",
    6: "\nSelect the subset identifier.",
    7: "Which layer the map should be applied to?",
    
}


class AbstractInformationExtractor(ABC):
    """Abstract class for the annotation-specific Extractor module.
    Blueprint for the extractor classes that need to be implemented
    for a new annotation type.

    Args:
        ABC (class): AbstractBaseClass object.
    """

    def __init__(self) -> NoReturn:
        super().__init__()
        self.input_handler = InputHandler()

    def set_collector(self, data_collector: Collector) -> NoReturn:
        """Sets collector to extract the annoations from.

        Args:
            data_collector (Collector): Container for annotation files.
        """
        self.collector = data_collector

    @abstractmethod
    def process_information(self) -> List[Any]:
        """Prepare the files in order to extract
        the information.

        Raises:
            NotImplementedError: Has to be method of
            Extractor.

        Returns:
            List[Any]: List of information.
        """
        raise NotImplementedError

    @abstractmethod
    def extract(self) -> List[Tuple[str, Any]]:
        """Main method to read the data from
        the files.

        Raises:
            NotImplementedError: Has to be implemented
            for any Extractor.

        Returns:
            List[Tuple[str, Any]]: Listing of the path of the file
            and the extracted annotation or information.
        """
        raise NotImplementedError
    
    
    @staticmethod
    def show_attributes(information_list: List[Dict[str, str]]) -> Dict[int, List[str]]:
        """Displays the attributes in the command line with an assigned id.
        This id can be selected to provide an attribute (i.e. folder name or xml instance name)
        which is used to assigne a label.

        Args:
            information_list (List[Dict[str, str]]): Extracted information from files.

        Returns:
            Dict[int, List[str]]: Mapping from id to attribute.
        """
        console_dict = {}
        for entry in information_list:
            for key in entry:
                if key in console_dict:
                    console_dict[key].add(entry[key])
                else:
                    console_dict[key] = set([entry[key]])
        print(PRINTOUTS[0])
        show_data(console_dict, console_dict.keys(), key2row=False)
        return console_dict

    def select_attribute(
        self,
        attribute_keys: List[int],
        printout_nr: int,
        extended_choice: Optional[List[str]] = None,
        dynamic_input: Optional[str] = "",

    ) -> int:
        """Select an id assigned to attribute for further processing.

        Args:
            attribute_keys (List[int]): Names of attributes.
            printout_nr (int): Number of command line statement.
            extended_choice (Optional[List[str]], optional): Additional command line input
            that can be chosen apart from attributes. Defaults to [].

        Returns:
            int: chosen attribute id.
        """
        attribute = self.input_handler.expect_input(
            PRINTOUTS[printout_nr] if dynamic_input == "" else PRINTOUTS[printout_nr].format(dynamic_input),
            choices=attribute_keys if extended_choice is None else attribute_keys + extended_choice,
        )
        self.input_handler.check_input(
            attribute,
            [str(key) for key in attribute_keys]
            if extended_choice is None
            else [str(key) for key in attribute_keys] + extended_choice,
        )
        return convert_type(attribute, int)

