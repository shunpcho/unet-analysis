from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from unet_analysis.configs.train_cfg import PairingKeyWords

MIN_DETECTORS_NUM = 2


def _load_img_np(path: Path) -> npt.NDArray[np.uint8]:
    """Load image from path and convert to numpy array."""
    return np.array(Image.open(path))


def load_img(pairing_words: PairingKeyWords | None, noisy: bool = False) -> Callable[[Path], npt.NDArray[np.uint8]]:
    """Return function to get image loading function based on pairing configuration.

    When noisy images are loaded, set noisy to True to replace the clean keyword with the noisy keyword in the
    filename. If noisy keyword is not given, add Gaussian noise to the clean image to create the noisy image.


    This function creates an appropriate image loader based on the detector configuration:

    1. **Dual detectors case**: When pairing_words.detector contains 2+ elements, loads
       images from both detectors as grayscale, expands each to (H, W, 1), and
       concatenates them channel-wise to create a (H, W, 2) array.

    2. **Single detector case**: When pairing_words.detector contains exactly 1 element,
       loads the detector image as grayscale and adds a channel dimension to create
       a (H, W, 1) array.

    3. **Color image case**: When pairing_words is None or has no detector configuration,
       loads the image as RGB color image, resulting in a (H, W, 3) array.

    Args:
        pairing_words: Configuration containing detector keywords for multi-detector setups.
                      If None, defaults to RGB color image loading.
        noisy: If True, replaces the clean keyword with the noisy keyword in the filename or adds Gaussian noise.

    Returns:
        A function that takes a Path and returns the appropriate image array based on
        the detector configuration.
    """
    "If pairing_words is given and has 2+ detectors, load images from both detectors as grayscale and concatenate them."
    if pairing_words is not None:
        "If 2+ detectors, load images from the top two detectors."
        if pairing_words.detector is not None and len(pairing_words.detector) >= MIN_DETECTORS_NUM:
            det0 = pairing_words.detector[0]
            det1 = pairing_words.detector[1]

            def _load_dual_detector(path: Path) -> npt.NDArray[np.uint8]:
                """Load images from both detectors, and concatenate channel-wise."""
                if noisy:
                    path = path.with_name(path.name.replace(pairing_words.clean, pairing_words.noisy))
                det0_path = path
                det1_path = path.with_name(path.name.replace(det0, det1))
                img_det0 = _load_img_np(det0_path)
                img_det1 = _load_img_np(det1_path)

                "Expand dimensions: (H, W) -> (H, W, 1)"
                img_det0 = np.expand_dims(img_det0, axis=-1)
                img_det1 = np.expand_dims(img_det1, axis=-1)

                return np.concatenate([img_det0, img_det1], axis=-1)

            return _load_dual_detector

        if pairing_words.detector is not None and len(pairing_words.detector) == 1:

            def _load_single_detector(path: Path) -> npt.NDArray[np.uint8]:
                """Load image from single detector, and add channel dimension."""
                if noisy:
                    path = path.with_name(path.name.replace(pairing_words.clean, pairing_words.noisy))
                img_det = _load_img_np(path)
                "Expand dimensions: (H, W) -> (H, W, 1)"
                return np.expand_dims(img_det, axis=-1)

            return _load_single_detector

    if noisy:
        "If noisy keyword is not given, return function to add Gaussian noise to clean image."
        return _add_gaussian_noise_factory(noise_sigma=0.08)

    "Default: Load color image as RGB (H, W, 3)."
    return _load_img_np


def _add_gaussian_noise_factory(noise_sigma: float) -> Callable[[Path], npt.NDArray[np.uint8]]:
    """Create a function that adds Gaussian noise to a clean image.

    This factory function returns a pairing function that, when called with a Path,
    loads the clean image using the provided data loading function and adds Gaussian
    noise with the specified standard deviation.

    Args:
        noise_sigma: Standard deviation of the Gaussian noise to be added.

    Returns:
        A function that takes a Path, loads the clean image, and returns the noisy image.
    """

    def _add_gaussian_noise(path: Path) -> npt.NDArray[np.uint8]:
        """Load clean image from path and add Gaussian noise."""
        clean_img = _load_img_np(path)
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_sigma * 255, clean_img.shape).astype(np.int16)
        noisy_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_img

    return _add_gaussian_noise
