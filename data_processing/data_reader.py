"""Read data content from collected data sources."""

from typing import Any
from typing import NoReturn
from typing import Tuple
from typing import Optional

from data_processing.helpers.input_handler import InputHandler
from data_processing.abstract_classes.abstract_reader import AbstractReader

from data_processing.extractors.path_extractor import PathExtractor


input_handler = InputHandler()

MODES = {
    "annotations": {
        "png": "id_str",
        "jpg": "id_str",
        "jpeg": "id_str",
        "xml": "xml",
        "json": "json",
        "txt": "txt",
        "png_mask": "mask",
    },
    "data": {
        "png": "image",
        "jpg": "image",
        "jpeg": "image",
    },
}

EXTRACTORS = {"id_str": PathExtractor()}

PRINTOUTS = {
    0: "Based on the file type(s) '{0}' no information extractor "
    "could have been instanciated due to the following reasons:\n"
    "1. Add the selected file types to the Reader MODES\n"
    "2. Implement a new information extractor class that "
    "can handle the '{0}' file type(s)"
}

PNG_TYPES = ["masks", "image data"]

FILE_PLACEHOLDER = "<file>"


class Reader(AbstractReader):
    """Reads the annotations based on
    implemented extractor class.

    Args:
        AbstractReader (class): Abstract class with general
        data selection implementation.
        category (str): Read data or annotations.
        out_dir (Optional[str], optional): Ouptput to transfer data if needed.
        Defaults to None.
        transfer_files (Optional[bool], optional): Transfer image files to
        given output directory. Defaults to False.

    Raises:
        KeyError: Raises if not supported category is chosen.
    """

    def __init__(
        self,
        category: str,
        out_dir: Optional[str] = None,
        transfer_files: Optional[bool] = False,
    ) -> NoReturn:
        if category not in MODES:
            raise KeyError(
                "'%s' is not in the supported categories %s."
                % (category, list(MODES.keys())),
            )
        self.category = category
        self.out_dir = out_dir
        self.transfer_files = transfer_files
        AbstractReader.__init__(self)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, Any]:
        return self.data[index]

    def _set_extractor(self) -> NoReturn:
        """Based on file type an implemented extractor class is chosen."""
        if not all(
            file_format in MODES[self.category] for file_format in self.file_formats
        ):
            raise KeyError(PRINTOUTS[0].format(",".join(self.file_formats)))
        self.modes = [
            MODES[self.category][file_format]
            for file_format in self.file_formats
        ]
        self.information_extractors = [EXTRACTORS[mode] for mode in self.modes]

    def read(self) -> NoReturn:
        """Reads the information from given files
        by calling suitable extractor.
        """
        self._set_extractor()
        for extractor in self.information_extractors:
            extractor.set_collector(self.data_collection)
            # extract the actual data
            # from collected files
            self.data += extractor.extract()
            if isinstance(extractor, PathExtractor) and self.transfer_files:
                self.data = extractor.transfer_files(
                    *list(zip(*self.data)), self.out_dir
                )
