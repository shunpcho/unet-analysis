import numpy as np
import numpy.typing as npt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from unet_analysis.configs.train_cfg import TrainConfig
from unet_analysis.utils.metrics import ComputeMetrics


class Trainer:
    """Trainer for training UNet model."""

    def __init__(
        self,
        train_config: TrainConfig,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader[
            tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]
        ],
        val_loader: DataLoader[
            tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]
        ],
        device: torch.device,
    ) -> None:
        """Initialize Trainer instance.

        Args:
        train_config: TrainConfig for training process.
        model: Model for training.
        optimizer: Optimizer for training.
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
        device: Device for training.
        """
        self.train_config = train_config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def loop(self, train: bool = True) -> dict[str, float]:
        """Run training loop."""
        self.model.train()
        dataloader = self.train_loader if train else self.val_loader
        steps = len(dataloader)
        step_losses: dict[str, float] = {}
        loss_fn = nn.MSELoss().to(self.device)

        for step, batch in enumerate(dataloader, 1):
            clean, noisy = batch
            clean = clean.to(self.device)
            noisy = noisy.to(self.device)

            if train:
                self.optimizer.zero_grad()

            outputs = self.model(noisy)
            loss = loss_fn(outputs, clean)

            if train:
                loss.backward()
                self.optimizer.step()

            step_losses["Loss"] = step_losses.get("Loss", 0) + loss.item()
            self._accumulate_metrics(outputs, clean, step_losses)

            self._print_progress(step, steps, loss)

        return {key: value / steps for key, value in step_losses.items()}

    def train_step(self) -> dict[str, float]:
        """Run one training step."""
        return self.loop(train=True)

    def val_step(self) -> dict[str, float]:
        """Run one validation step."""
        with torch.no_grad():
            step_losses = self.loop(train=False)
        return step_losses

    @staticmethod
    def _print_progress(step: int, max_iterations: int, loss: float | torch.Tensor) -> None:
        """Print training progress bar."""
        if max_iterations <= 0:
            return

        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
        progress = step / max_iterations
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "=" * filled + "." * (bar_width - filled)
        end = "\n" if step == max_iterations else "\r"
        print(
            f"[{bar}] {step:>3}/{max_iterations} ({progress * 100:6.2f}%) Loss: {loss_value:.4f}",
            end=end,
            flush=True,
        )

    def _accumulate_metrics(self, outputs: torch.Tensor, clean: torch.Tensor, step_losses: dict[str, float]) -> None:
        """Accumulate metrics of MSE, PSNR and SSIM."""
        with torch.no_grad():
            compute_metrics = ComputeMetrics().to(self.device)
            metrics = compute_metrics(outputs, clean)
        step_losses["MSE"] = step_losses.get("MSE", 0.0) + metrics["MSE"]
        step_losses["PSNR"] = step_losses.get("PSNR", 0.0) + metrics["PSNR"]
        step_losses["SSIM"] = step_losses.get("SSIM", 0.0) + metrics["SSIM"]
