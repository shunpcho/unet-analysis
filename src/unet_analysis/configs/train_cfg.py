from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Self, TypedDict, Unpack

import torch


class PairingKeyWords(NamedTuple):
    """Dataset Config for pairing clean and noisy images.

    clean: Keyword to identify clean images in filename.
    noisy: Keyword to identify noisy images in filename. (Optional)
    detector: List of detector keywords to filter images. (Optional)
    """

    clean: str
    noisy: str | None = None
    detector: list[str] | None = None


class _TrainConfigKwargs(TypedDict, total=False):
    dataset_config: PairingKeyWords | None
    seed: int
    crop_size: int | tuple[int, int] | None
    batch_size: int
    iterations: int
    interval: int
    learning_rate: float
    optimizer: str
    normalize_type: str
    loss_fn: str
    model_arch: str
    output_dir: Path
    device: torch.device


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Train Config for run args.

    Args:
        dataset_config: Keywords for identifying or filtering images.
        seed: Seed for random operations.
        crop_size: Size for random cropping of images.
        batch_size: Batch size for learning.
        iterations: Iterations for learning.
        interval: Interval for logging or saving during training.
        learning_rate: Learning rate for optimizer.
        optimizer: Optimizer for learning.
        normalize_type: Normalization type for model layers.
        loss_fn: Loss function for learning.
        model_arch: Model architecture for learning.
        output_dir: Path of output directory for saving model and logs.
        device: Device for training. (e.g., "cuda" or "cpu")

    """

    dataset_config: PairingKeyWords | None
    seed: int
    crop_size: int | tuple[int, int] | None
    batch_size: int
    iterations: int
    interval: int
    learning_rate: float
    optimizer: str
    normalize_type: str
    loss_fn: str
    model_arch: str
    output_dir: Path
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @classmethod
    def from_optional_kwargs(cls, **kwargs: Unpack[_TrainConfigKwargs]) -> Self:
        """Create a TrainConfig instance from given keyword arguments.

        This method allows creating a TrainConfig instance by providing only
        the desired parameters, while the rest will take default values.

        Args:
            **kwargs: Keyword arguments corresponding to TrainConfig fields.

        Returns:
            TrainConfig instance created from given keyword arguments.
        """
        return cls(**{key: value for key, value in kwargs.items() if value is not None})  # pyright: ignore[reportArgumentType]
