"""Implementations needed for specific task."""

from abc import ABC
import json
import os
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Any
from typing import NoReturn

from PIL import Image
import torch
from torch.nn import Module
from torch.utils.data import Dataset

from model_training.annotation_handler import AnnotationHandler
from model_training.transformation import image_preprocessing

CATEGORY_MAP_NAME = "categories.json"


class AbstractDataSet(ABC, Dataset):
    """Guideline for the data set object. A new to implement data set object
    can inherit from it to obtain basic default functionalities.

    Args:
        data (List[str]): Data file paths.
        annotation (List[str]): File paths to assigned annotations.
        transformation (Module, optional): Augmentation pipeline applied to the data.
        Defaults to None.
        dims (Tuple[int, int], optional): Desired image size.
        Defaults to None.
    """

    def __init__(
        self,
        data: List[str],
        annotation: List[str],
        transformation: Module = None,
        dims: Tuple[int, int] = None,
    ) -> NoReturn:

        super().__init__()
        self.data = data
        self.annotation = annotation
        self.transformation = transformation
        self.dims = dims
        self.annotation_handler = AnnotationHandler()

    def read_image_data(self, path: str) -> Image.Image:
        """Load PIL Image from path and transfrom to RGB.

        Args:
            path (str): File path with supported file format.

        Returns:
            Image.Image: Image object from file.
        """
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def read_annotation(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Read annotation object from json file.

        Args:
            path (str): Json file path.

        Raises:
            IOError: Raises is file can not be loaded properly.

        Returns:
            Dict[str, Dict[str, Any]]:
        """
        with open(path, "r") as file_handle:
            try:
                return json.load(file_handle)
            except IOError as error:
                raise IOError(
                    "Not able to read annotation from '%s'." % path
                ) from error


class ImageClassificationDataSet(AbstractDataSet):
    """Data set for streaming images and labels for tasks like image
    classification, etc.

    Args:
        data (List[str]): Data file paths.
        annotation (List[str]): File paths to assigned annotations.
        transformation (Module, optional): Augmentation pipeline applied to the data.
        Defaults to None.
        dims (Tuple[int, int], optional): Desired image size.
        Defaults to None.
        kwargs: (Optional[Any], optional): Any parameters required by this
        specific data set.
    """

    def __init__(
        self,
        data: List[str],
        annotation: List[str],
        transformation: Module = None,
        dims: Tuple[int, int] = None,
        **kwargs: Optional[Any]
    ) -> NoReturn:
        super().__init__(data, annotation, transformation=transformation, dims=dims)
        self.preprocessing = image_preprocessing()

    def __len__(self) -> int:
        """Defines the length of the data set object.

        Returns:
            int: Number of data files.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Getter function to obtain data-annotation pair.

        Args:
            index (int): Position in the data set object.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data sample with assigned class.
        """
        sample = self.read_image_data(self.data[index])
        annotation = self.read_annotation(self.annotation[index])
        _, label = self.annotation_handler.extract(annotation)

        label = torch.tensor(label)
        if self.dims:
            sample = sample.resize(self.dims)
        if self.transformation:
            sample = self.transformation(sample)
        sample = self.preprocessing(sample)
        return sample, label


class TripletDataSet(AbstractDataSet):
    """Data set for triples constructed from streamed images.

    Args:
        data (List[str]): Data file paths.
        annotation (List[str]): File paths to assigned annotations.
        transformation (Module, optional): Augmentation pipeline applied to the data.
        Defaults to None.
        dims (Tuple[int, int], optional): Desired image size.
        Defaults to None.
        kwargs: (Optional[Any], optional): Any parameters required by this
        specific data set.
    """

    def __init__(
        self,
        data: List[str],
        annotation: List[str],
        transformation: Module = None,
        dims: Tuple[int, int] = None,
        **kwargs: Optional[Any]
    ) -> NoReturn:
        super().__init__(data, annotation, transformation=transformation, dims=dims)
        self.preprocessing = image_preprocessing()
        category_map = kwargs.get("category_map", None)

        self.categories = (
            {} if category_map is None else json.load(open(category_map, "r"))
        )

    def _get_triple_member(
        self, anchor_label: int, type: str, category: Optional[int] = None
    ) -> str:
        """Construct triples by randomly select either positive or negative
        sample. If a category is given, triples are constructed by samples
        that match this category.

        Args:
            anchor_label (int): Label of the anchor to decide on positive
            and negative sample.
            type (str): Either select a positive or a negative sample.
            Options: "pos"/"neg".
            category (Optional[int], optional): Additional parameter to
            define a subset of samples. Defaults to None.

        Raises:
            KeyError: Type is not "pos" or "neg".
            AssertionError: Raises if no remaining samples for triple are given.

        Returns:
            str: Name of annotation file.
        """
        triple_member = ["pos", "neg"]
        if type not in triple_member:
            raise KeyError("Type of triple member has to be in %s" % triple_member)

        # restrict to category if given
        if category:
            samples, labels = zip(*self._get_category_samples(category))
        else:
            samples, labels = self.data, self.annotation
        labels = list(map(self.read_annotation, labels))
        _, labels = zip(*list(map(self.annotation_handler.extract, labels)))

        samples = (
            [sample for sample, label in zip(samples, labels) if label == anchor_label]
            if type == "pos"
            else [
                sample
                for sample, label in zip(samples, labels)
                if label != anchor_label
            ]
        )
        if len(samples) == 0:
            raise AssertionError("No samples to choose from for triple!")

        member = random.choice(samples)
        return member

    def _get_category_samples(self, category: int) -> List[Tuple[str, str]]:
        """Get list of sample annotation pairs that meet the category
        criteria.

        Args:
            category (int): Subset indicator for further split
            of the data.

        Returns:
            List[Tuple[str, str]]: Choosen sample matching
            the category restrictions.
        """
        return [
            (sample, label)
            for sample, label in zip(self.data, self.annotation)
            if self.categories[os.path.basename(sample)] == category
        ]

    def __len__(self) -> int:
        """Size of the data set to make it
        iterable.

        Returns:
            int: Number of data samples contained by
            the data set.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Selects positive and negative sample in regards to anchor label.

        Args:
            index (int): Current index of the data set.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Triple of images and anchor label.
        """
        anchor = self.read_image_data(self.data[index])
        annotation = self.read_annotation(self.annotation[index])

        _, anchor_label = self.annotation_handler.extract(annotation)
        pos = self.read_image_data(
            self._get_triple_member(
                anchor_label,
                "pos",
                self.categories.get(os.path.basename(self.data[index]), None),
            )
        )
        neg = self.read_image_data(
            self._get_triple_member(
                anchor_label,
                "neg",
                self.categories.get(os.path.basename(self.data[index]), None),
            )
        )

        triple = []
        for item in [anchor, pos, neg]:
            if self.dims:
                item = item.resize(self.dims)
            item = self.preprocessing(item)
            if self.transformation:
                item = self.transformation(item)
            triple.append(item)
        anchor_label = torch.tensor(anchor_label)

        return triple, anchor_label
