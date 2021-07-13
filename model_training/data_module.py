"""Implementation of the general data set module."""

import os
from logging import getLogger
from typing import List
from typing import NoReturn

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn import Module

from model_training.data_set import ImageClassificationDataSet, TripletDataSet

logger = getLogger(__file__)

SPLIT_FOLDERS = ["train", "val", "test"]
DATA_FOLDER = "data"
ANNOTATION_FOLDER = "annotations"

DATA_SETS = {0: ImageClassificationDataSet, 1: TripletDataSet}


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        set_id: int,
        train_transforms: Module = None,
        val_transforms: Module = None,
        test_transforms: Module = None,
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        image_dim: List[int] = None,
        **kwargs
    ) -> NoReturn:
        """Standardized form for the handling of the data loading
        and providing to the model.

        Args:
            data_dir (str): Parent folder for train/val/test data.
            set_id (int): Selector for the data set to process the data.
            train_transforms (Module, optional): Augmentations applied to the train set. Defaults to None.
            val_transforms (Module, optional): Augmentations applied to the val set. Defaults to None.
            test_transforms (Module, optional): Augmentations applied to the test set. Defaults to None.
            train_batch_size (int, optional): Batch sizes used for the train set. Defaults to 1.
            val_batch_size (int, optional): Batch sizes used for the val set. Defaults to 1.
            test_batch_size (int, optional): Batch sizes used for the test set. Defaults to 1.
            image_dim (List[int], optional): Dimension of images as a standard for the training. Defaults to None.
        """
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            dims=tuple(image_dim),
        )

        self.data_dir = data_dir
        self.data_set = DATA_SETS[set_id]
        self._check_data_dir()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        # these kwargs can be used
        # to provide further specific
        # parameters
        self.kwargs = kwargs
        self.setup()

    def _check_data_dir(self) -> NoReturn:
        """Checks if all required folders
        are existent.

        Raises:
            ValueError: Not all required folders exist.
        """
        if not all(map(lambda x: x in os.listdir(self.data_dir), SPLIT_FOLDERS)):
            raise ValueError(
                "'%s' has to contain the folders %s", self.data_dir, SPLIT_FOLDERS
            )

    def setup(self) -> NoReturn:
        """Setup of the datasets for the different training stages."""
        self.data_dict = {}
        for split_folder in SPLIT_FOLDERS:
            self.data_dict[split_folder] = {}
            data_folder, annotation_folder = list(
                map(
                    lambda x, folder=split_folder: os.path.join(
                        self.data_dir, folder, x
                    ),
                    [DATA_FOLDER, ANNOTATION_FOLDER],
                )
            )
            data_files = self._collect_files(data_folder)
            annotation_files = list(
                map(
                    lambda x, folder=annotation_folder: os.path.join(
                        folder,
                        os.path.basename(x)
                        .replace(".png", ".json")
                        .replace(".jpg", ".json"),
                    ),
                    data_files,
                )
            )

            if split_folder == "train":
                self.train_set = self.data_set(
                    data_files,
                    annotation_files,
                    self.train_transforms,
                    self.dims,
                    **self.kwargs
                )
            elif split_folder == "val":
                self.val_set = self.data_set(
                    data_files,
                    annotation_files,
                    self.val_transforms,
                    self.dims,
                    **self.kwargs
                )
            if split_folder == "test":
                self.test_set = self.data_set(
                    data_files,
                    annotation_files,
                    self.test_transforms,
                    self.dims,
                    **self.kwargs
                )

    def train_dataloader(self) -> DataLoader:
        """Dataloader getter to stream the data for training from.

        Returns:
            DataLoader: Train Dataloader object.
        """
        return DataLoader(
            self.train_set, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """Dataloader getter to stream the data for validation from.

        Returns:
            DataLoader: Val Dataloader object.
        """
        return DataLoader(self.val_set, batch_size=self.val_batch_size)

    def test_dataloader(self) -> DataLoader:
        """Dataloader getter to stream the data for training from.

        Returns:
            DataLoader: Train Dataloader object.
        """
        return DataLoader(self.test_set, batch_size=self.test_batch_size)

    @staticmethod
    def _collect_files(directory: str) -> List[str]:
        """Collects all file path names from files
        in given directory.

        Args:
            directory (str): Directory path to
            collect filenames from.

        Returns:
            List[str]: File path for each file
            that is not an invisible file.
        """
        files = []
        for file_name in os.listdir(directory):
            if not file_name.startswith("."):
                files.append(os.path.abspath(os.path.join(directory, file_name)))
        logger.info("Collected %i files from '%s'", len(files), directory)
        return files
