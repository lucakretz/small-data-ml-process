"""Collection of functionality for data type and format conversion."""

from collections import MutableMapping
from typing import Any
from typing import Dict
from typing import List
from typing import Union
import numpy as np


def convert_type(obj: Any, output_type: object) -> Any:
    """Converts object to new given type if possible.

    Args:
        obj (Any): Object to transform.
        output_type (object): Type like int or str.

    Returns:
        Any: Object in with new type.
    """
    try:
        obj = output_type(obj)
    except ValueError:
        pass
    return obj


def flatten_dict(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively unzips dictionary
    to the pre last instance. Brings each key to first instance.

    Args:
        dictionary (Dict[str, Any]): Normal dictionary
        with any number of instances.

    Returns:
        Dict[Any]: Dictionary with only one instance.
    """
    items = []
    for key, value in dictionary.items():
        items.append((key, value))
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value).items())
    return dict(items)


def add_to_array(item: Union[int, float], array: Union[None, np.array]) -> np.array:
    """Adds new item to numpy array or creates new if empty.

    Args:
        item (Union[int, float]): New instance for array.
        array (Union[None, np.array]): Existing array to extend.

    Returns:
        np.array: Array with item on last position.
    """
    if array:
        return np.append(array, item)
    return np.array(item)


def extract_classes(
    annotation_list: List[Dict[int, Dict[str, Any]]],
    class_key: str,
    empty_annotation: str,
) -> List[str]:
    """Extracts list of classes from annotation format.

    Args:
        annotation_list (List[Dict[int, Dict[str, Any]]]): List with
        annotations in predefined format.
        class_key (str): Key that holds the class name.
        empty_annotation (str): Value to return if the annotation
        is empty.

    Returns:
        List[str]: List of extracted class names.
    """
    classes = []
    for annotation in annotation_list:
        if annotation:
            class_names = set()
            for annotation_dict in annotation.values():
                class_names.add(annotation_dict[class_key])
            class_list = sorted(list(class_names))
        else:
            class_list = empty_annotation
        classes.append(str(class_list))
    return classes
