import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


class PairedDataset(
    Dataset[tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]]
):
    """Dataset for paired clean and noisy images.

    Args:
        data_paths: Root directory of images.
        data_loading_fn: Function to load image from path.
        img_standardization_fn: Function to standardize image.
        pairing_fn: Function to get paired image paths of clean and noisy.
        data_augmentation_fn: Function to augment image.
        limit: if given, limit number of images to load.
        mode: "train" or "val" for separate dataset.

    """

    def __init__(
        self,
        data_path: Path | list[Path],
        data_loading_fn: Callable[[Path], npt.NDArray[np.uint8]],
        img_standardization_fn: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float32]],
        pairing_fn: Callable[[Path], npt.NDArray[np.uint8]],
        data_augmentation_fn: Callable[
            [npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
            tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
        ]
        | None = None,
        mode: Literal["train", "val"] = "train",
        limit: int | None = None,
    ) -> None:
        self.data_loading_fn = data_loading_fn
        self.img_standardization_fn = img_standardization_fn
        self.pairing_fn = pairing_fn
        self.data_augmentation_fn = data_augmentation_fn
        self.mode = mode

        if isinstance(data_path, Path):
            self.data_path = [data_path]
        else:
            self.data_path = data_path
        self.img_paths = self._load_image_paths()

        random.shuffle(self.img_paths)
        if limit is not None:
            self.img_paths = self.img_paths[:limit]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]:
        """Get paired clean and noisy images.

        Args:
            idx: Index of dataset item. If idx exceeds dataset length, it will be wrapped around using modulo.

        Returns:
            Tuple of clean and noisy images as numpy arrays.
                data shape: Trainable pytorch tensor (C, H, W).
                dtype: uint8 or float32 depending on data_loading_fn and img_standardization_fn.
        """
        clean_path = self.img_paths[idx % len(self.img_paths)]
        clean_img = self.data_loading_fn(clean_path)
        noisy_img = self.pairing_fn(clean_path)

        if self.data_augmentation_fn is not None:
            clean_img, noisy_img = self.data_augmentation_fn(clean_img, noisy_img)

        clean_img = self.img_standardization_fn(clean_img)
        noisy_img = self.img_standardization_fn(noisy_img)

        clean_img = hwc_to_chw(clean_img)
        noisy_img = hwc_to_chw(noisy_img)

        return clean_img, noisy_img

    def _load_image_paths(self) -> list[Path]:
        """Load train/val image paths by reading pre-created index list JSON files.

        Returns:
            List of image paths.

        Raises:
            FileNotFoundError: If indices file not found in data_path.
        """
        img_paths: list[Path] = []

        "File existence check and load indices files for each data_path."
        for data_path in self.data_path:
            indices_path = data_path / "indices" / f"{self.mode}_list.json"
            if not indices_path.exists():
                msg = f"Indices file not found: {indices_path}. Please create indices files."
                raise FileNotFoundError(msg)

            with indices_path.open(encoding="utf-8") as f:
                indices = json.load(f)
            img_paths += [Path(img_path) for img_path in indices]

        return img_paths


GRAYSCALE_DIM = 2
COLOR_DIM = 3


@overload
def hwc_to_chw(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...


@overload
def hwc_to_chw(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...


def hwc_to_chw(img: npt.NDArray[np.uint8] | npt.NDArray[np.float32]) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
    """Convert image from (H, W, C) to (C, H, W).

    If grayscale image with shape (H, W), add channel dimension and convert to (1, H, W).

    Args:
        img: Image as numpy array with shape (H, W, C) and dtype uint8 or float32.

    Returns:
        Image as numpy array with shape (C, H, W) and same dtype as input.

    Raises:
        ValueError: If input image does not have 2 or 3 dimensions.
    """
    if img.ndim == GRAYSCALE_DIM:
        img = img[np.newaxis, :, :]
    elif img.ndim == COLOR_DIM:
        img = np.transpose(img, (2, 0, 1))  # type: ignore[return-value]
    else:
        msg = f"Unsupported image shape: {img.shape}. Expected 2D (H, W) or 3D (H, W, C) array."
        raise ValueError(msg)
    return img
