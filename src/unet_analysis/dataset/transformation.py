from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

BATCH_ENTRY_LENGTH = 2


def compose_transformations(
    transforms: list[
        Callable[
            [npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
            tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
        ]
    ],
) -> Callable[
    [npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
    tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
]:
    """Compose a list of transformations into a single transformation function.

    Args:
        transforms: A list of transformation functions to be applied sequentially.

    Returns:
        A single transformation function that applies all the transformations in sequence.
    """

    def _fn(
        clean: npt.NDArray[np.uint8 | np.float32], noisy: npt.NDArray[np.uint8 | np.float32]
    ) -> tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]]:
        """Apply composed transformations to a pair of clean and noisy images."""
        for transform in transforms:
            clean, noisy = transform(clean, noisy)
        return clean, noisy

    return _fn


def random_crop(
    crop_size: int | tuple[int, int] | None, seed: int = 42
) -> Callable[
    [npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
    tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
]:
    """Create a function to perform random crop.

    Args:
        crop_size: Size of the crop. If int, crop will be square with (crop_size, crop_size).
        seed: Seed for random operations.

    Returns:
        A function that takes a pair of clean and noisy images, and returns a randomly cropped pair of images.
    """
    "If crop size is not given, return identity function."
    if crop_size is None:

        def no_crop(
            clean: npt.NDArray[np.uint8 | np.float32], noisy: npt.NDArray[np.uint8 | np.float32]
        ) -> tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]]:
            return clean, noisy

        return no_crop

    "If crop size is given, return function to perform random crop."
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    crop_h, crop_w = crop_size

    def _random_crop(
        clean: npt.NDArray[np.uint8 | np.float32], noisy: npt.NDArray[np.uint8 | np.float32]
    ) -> tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]]:
        """Perform random crop on a pair of clean and noisy images.

        Args:
            clean: Clean image as a NumPy array.
            noisy: Noisy image as a NumPy array.

        Returns:
            A tuple of randomly cropped clean and noisy images.

        Raises:
            ValueError: If crop size is larger than image size.
        """
        rng = np.random.default_rng(seed)
        h, w = clean.shape[:2]
        if h < crop_h or w < crop_w:
            msg = f"Crop size {crop_size} is larger than image size {(h, w)}."
            raise ValueError(msg)
        "Randomly select top-left corner of the crop."
        top: int = int(rng.integers(0, int(h - crop_h + 1)))
        left: int = int(rng.integers(0, int(w - crop_w + 1)))

        "Crop clean and noisy images using the same coordinates."
        clean_crop = clean[top : top + crop_h, left : left + crop_w]
        noisy_crop = noisy[top : top + crop_h, left : left + crop_w]
        return clean_crop, noisy_crop

    return _random_crop


def collate_fn(
    batch: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function to stack clean and noisy images into batches.

    Args:
        batch: A list of NumPy tuples where each tuple contains a clean and a noisy image.

    Returns:
        A tuple of two NumPy arrays: (clean_batch, noisy_batch).
    """
    clean_imgs: list[torch.Tensor] = []
    noisy_imgs: list[torch.Tensor] = []

    for clean_img, noisy_img in batch:
        clean_tensor = torch.from_numpy(clean_img).float()
        noisy_tensor = torch.from_numpy(noisy_img).float()

        clean_imgs.append(clean_tensor)
        noisy_imgs.append(noisy_tensor)

    clean_batch = torch.stack(clean_imgs, dim=0)
    noisy_batch = torch.stack(noisy_imgs, dim=0)

    return clean_batch, noisy_batch


def standardize_img(
    mean: tuple[float, ...], std: tuple[float, ...]
) -> Callable[[npt.NDArray[np.uint8 | np.float32]], npt.NDArray[np.float32]]:
    """Create a function to standardize images.

    Args:
        mean: A tuple of mean values for each channel.
        std: A tuple of standard deviation values for each channel.

    Returns:
        A function that takes an image as a NumPy array and returns a standardized image.
    """
    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    max_val = np.float32(255.0)

    def _standardize_img(img: npt.NDArray[np.uint8 | np.float32]) -> npt.NDArray[np.float32]:
        """Standardize an image using the provided mean and std.

        Args:
            img: An image as a NumPy array.

        Returns:
            A standardized image as a NumPy array.
        """
        img = img.astype(np.float32)
        standardized_img = (img - mean_array * max_val) / (std_array * max_val)

        return standardized_img.astype(np.float32)

    return _standardize_img
