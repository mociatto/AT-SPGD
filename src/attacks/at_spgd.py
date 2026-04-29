from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATSPGD(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha_f: float = 2 / 255,
        alpha_x: Optional[float] = None,
        steps: int = 10,
        K: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < K <= 1.0:
            raise ValueError("K must be in the interval (0, 1].")
        self.model = model
        self.eps = eps
        self.alpha_f = alpha_f
        self.alpha_x = alpha_x if alpha_x is not None else alpha_f
        self.steps = steps
        self.K = K

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x0 = images.detach().clone()
        x = x0.clone()

        for _ in range(self.steps):
            x = x.detach().requires_grad_(True)
            logits = self.model(x)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            gradient = x.grad
            if gradient is None:
                break

            spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
            gradient_spectrum = torch.fft.fft2(gradient, dim=(-2, -1), norm="ortho")
            magnitude = torch.abs(spectrum)
            phase = torch.angle(spectrum)

            magnitude_gradient = (
                spectrum.real * gradient_spectrum.real
                + spectrum.imag * gradient_spectrum.imag
            ) / (magnitude + 1e-8)

            flat_gradient = torch.abs(magnitude_gradient).view(magnitude_gradient.size(0), -1)
            threshold = torch.quantile(flat_gradient, 1.0 - self.K, dim=1)
            mask = (
                torch.abs(magnitude_gradient)
                >= threshold.view(-1, 1, 1, 1)
            ).to(magnitude_gradient.dtype)

            magnitude = magnitude + self.alpha_f * (magnitude_gradient * mask).sign()
            filtered_spectrum = magnitude * torch.exp(1j * phase)
            x = torch.fft.ifft2(filtered_spectrum, dim=(-2, -1), norm="ortho").real

            x = torch.clamp(x, 0.0, 1.0)
            x = x0 + torch.clamp(x - x0, -self.eps, self.eps)
            x = torch.clamp(x, 0.0, 1.0)

        return x.detach()
