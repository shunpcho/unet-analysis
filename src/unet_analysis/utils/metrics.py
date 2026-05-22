from typing import Literal

import torch
import torch.nn.functional as f
from torch import nn


class ComputeMetrics(nn.Module):
    """Compute metrics of MSE, PSNR and SSIM."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.psnr = PSNRLoss()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Compute MSE, PSNR and SSIM metrics."""
        mse = self.mse(pred, target).item()
        psnr = self.psnr(pred, target).item()
        ssim = self.ssim(pred, target).item()
        return {"MSE": mse, "PSNR": psnr, "SSIM": ssim}


class PSNRLoss(nn.Module):
    """Implement PSNR loss."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        if mse_loss == 0:
            return torch.tensor(float("inf"))
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
        return psnr


class SSIMLoss(nn.Module):
    """Implement SSIM loss.

    - window: avg_pool2d with kernel_size=window_size
    - c1 = 0.01**2, c2 = 0.03**2  (assumes images in [0, 1])
    - returns: 1 - mean(ssim_map)
    """

    def __init__(self, window_size: int = 11, eps: float = 1e-12, reduction: Literal["mean", "none"] = "mean") -> None:
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        if reduction not in {"mean", "none"}:
            msg = "reduction must be 'mean' or 'none'"
            raise ValueError(msg)
        self.reduction = reduction

    def _compute_ssim_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the SSIM map for two images."""
        ws = self.window_size
        pad = ws // 2

        mu1 = f.avg_pool2d(pred, ws, padding=pad)
        mu2 = f.avg_pool2d(target, ws, padding=pad)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = f.avg_pool2d(pred * pred, ws, padding=pad) - mu1_sq
        sigma2_sq = f.avg_pool2d(target * target, ws, padding=pad) - mu2_sq
        sigma12 = f.avg_pool2d(pred * target, ws, padding=pad) - mu1_mu2

        c1 = 0.01**2
        c2 = 0.03**2

        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        return num / (den + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (N, C, H, W), float, range [0,1] assumed
        ssim_map = self._compute_ssim_map(pred, target)
        ssim = ssim_map.mean(dim=(-1, -2, -3))

        if self.reduction == "mean":
            ssim = ssim.mean()
            return 1.0 - ssim
        else:
            # Return (N,)
            return 1.0 - ssim
