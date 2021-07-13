"""Interface module to handle the previously created annotations."""

from typing import Tuple
from typing import Dict
from typing import Any
from typing import Optional

from PIL import Image

from data_processing.schemas import ANNOTATION_ENCODING, ANNOTATION_STR, ANNOTATION_TYPE


class AnnotationHandler:
    def extract(self, dict_object: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """

        Args:
            dict_object (Dict[str, Dict[str, Any]]): Annotation object
            produced by preprocessing pipeline.

        Raises:
            TypeError: For a label annotation more
            than one label per image is not allowed.
            KeyError: Annotation with unsupported type.

        Returns:
            Dict[str, Any]: Annotation object for each task.
        """
        extraction = []
        for key in dict_object:
            if dict_object[key][ANNOTATION_TYPE] == "label":
                extraction.append(self._handle_label(dict_object[key]))
                if len(dict_object) > 1:
                    raise TypeError(
                        "Label annotation with more "
                        "than one label is ambigous!"
                    )
                return extraction[0]
            elif dict_object[key][ANNOTATION_TYPE] == "bndbox":
                pass
            elif dict_object[key][ANNOTATION_TYPE] == "polygon":
                pass
            elif dict_object[key][ANNOTATION_TYPE] == "mask":
                pass
            else:
                raise KeyError(
                    "Annotation type '%s' not supported.", dict_object[key]["type"]
                )

    @staticmethod
    def _handle_label(info: Dict[str, Any]) -> Tuple[str, Any]:
        """Extracts annotation type and the actual annotation
        from an annotation file.

        Args:
            info (Dict[str, Any]): Content of annotation file
            produced by the preprocessing pipeline.

        Returns:
            Tuple[str, Any]: Type and assigned annotation object.
        """
        return info[ANNOTATION_STR], info[ANNOTATION_ENCODING]

    @staticmethod
    def resize(
        image: Image.Image,
        dim: Tuple[int, int],
        annotation_object: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        """Resize an image and if necessary the corresponding annotation.

        Args:
            image (Image.Image): Image object to resize.
            dim (Tuple[int, int]): New dimension of the image.
            annotation_object (Optional[Dict[str, Any]], optional): [description]. Defaults to None.

        Returns:
            Image.Image: Image in new shape.
        """
        if not annotation_object:
            return image.resize(dim)
