"""Script to define the applied augmentations and image preprocessing steps."""

from typing import Tuple
from typing import Optional

from torchvision import transforms

NORMALIZATION_FAKTORS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def image_preprocessing() -> transforms.Compose:
    """Standard preprocessing for images before training.

    Returns:
        transforms.Compose: Normalization and tensor conversion.
    """
    pipe = [
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_FAKTORS),
    ]
    return transforms.Compose(pipe)


def get_augmentations(
    image_dim: Optional[Tuple[int, int]] = None
) -> transforms.RandomChoice:
    """Augmentation pipeline for training.

    Args:
        image_dim (Tuple[int, int]): Desired image dimension.

    Returns:
        transforms.RandomChoice: Random selection of data augmentation
        function.
    """
    pipe = [
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(),
    ]
    return transforms.RandomChoice(pipe)
