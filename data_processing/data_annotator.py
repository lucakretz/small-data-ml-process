"""Class to produce the unique form for annotation files."""

import os
import json
from typing import Optional
from typing import NoReturn

from schema import SchemaError
from data_processing.data_reader import Reader
from data_processing.schemas import SCHEMAS
from data_processing.schemas import ANNOTATION_TYPE
from data_processing.schemas import ANNOTATION_STR
from data_processing.schemas import ANNOTATION_ENCODING

from utils.io_utils import mkdirr

PRINTOUTS = {0: "Ambiguous formats were found in the annotations: {}"}
ANNOTATIONS_DIR = "annotations"


class Annotator:
    """Module to transform the extracted annotations to
    a predefined structure and produce json files.

    Args:
        reader_object (Reader): Reader containing assignement from file to annotation.
    """

    def __init__(self, reader_object: Reader) -> None:
        self.annotations = reader_object

    @staticmethod
    def check_annotations(annotations: Reader) -> str:
        """Checks with annotation type is given based on schema.

        Args:
            annotations (Reader): List of annotation extracted from files.

        Raises:
            SchemaError: Schema of given annotation does not match a
            pre-defined schema.

        Returns:
            str: Unique annotation format.
        """
        formats = set()
        for obj in SCHEMAS:
            if all(SCHEMAS[obj].is_valid(annotation) for _, annotation in annotations):
                formats.add(obj)
        if len(formats) == 1:
            return list(formats)[0]
        raise SchemaError(PRINTOUTS[0].format(list(formats)))

    @staticmethod
    def check_output_folder(directory: str) -> str:
        """Creates the given folder to produce annotation files.

        Args:
            directory (str): Output folder directory.

        Returns:
            str: Directory or extended version.

        Raises:
            FileExistsError: Output directory contains files yet.
        """
        directory = os.path.join(directory, ANNOTATIONS_DIR)
        if not os.path.exists(directory):
            mkdirr(directory)
        return directory

    def produce_annotation_files(
        self, output_folder: str, multi_files: Optional[bool] = True
    ) -> NoReturn:
        """Produces json files to make annotations in pre defined format persistent.

        Args:
            output_folder (str): Folder to store files.
            multi_files (Optional[bool], optional): One file per annotation
            or one file for all. Defaults to True.
        """
        self.annotation_format = self.check_annotations(self.annotations)
        output_folder = self.check_output_folder(output_folder)
        # produces one file per annotation
        if multi_files:
            for file_path, annotation_list in self.annotations:
                annotation_file_path = os.path.join(
                    output_folder,
                    os.path.basename(file_path).replace(
                        file_path.split(".")[-1], "json"
                    ),
                )
                if not isinstance(annotation_list, list):
                    annotation_list = [annotation_list]
                with open(annotation_file_path, "w") as file_handle:
                    json.dump(
                        {
                            idx: {
                                ANNOTATION_TYPE: self.annotation_format,
                                ANNOTATION_STR: annotation[0],
                                ANNOTATION_ENCODING: annotation[1],
                            }
                            for idx, annotation in enumerate(annotation_list)
                        },
                        file_handle,
                    )
        # produces one file with all annotations
        else:
            annotation_file_content = {}
            for file_path, annotation in self.annotations:
                annotation_file_content[file_path] = {
                    self.annotation_format: annotation
                }
            with open(
                os.path.join(output_folder, "annotations.json"), "w"
            ) as file_handle:
                json.dump(annotation_file_content, file_handle)
