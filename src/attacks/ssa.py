from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSA(nn.Module):
    """
    Spectrum Simulation Attack (ECCV 2022).
    Adapts frequency-domain noise augmentation using native PyTorch FFT.
    """
    def __init__(self, model: nn.Module, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, N: int = 20, rho: float = 0.5) -> None:
        super().__init__()
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.N = N
        self.rho = rho

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x0 = images.clone().detach()
        x = x0.clone().detach()
        momentum = torch.zeros_like(x)

        for _ in range(self.steps):
            x.requires_grad_(True)
            grad_accum = torch.zeros_like(x)
            
            for _ in range(self.N):
                # 1. Transform to frequency domain
                spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
                amp = torch.abs(spectrum)
                phase = torch.angle(spectrum)
                
                # 2. Spectrum Simulation: Add uniform noise scaled by amplitude
                noise = (torch.rand_like(amp) - 0.5) * 2.0 * self.rho * amp
                simulated_amp = amp + noise
                
                # 3. Inverse transform back to spatial domain
                simulated_spectrum = simulated_amp * torch.exp(1j * phase)
                x_sim = torch.fft.ifft2(simulated_spectrum, dim=(-2, -1), norm="ortho").real
                
                # 4. Compute gradients
                logits = self.model(x_sim)
                loss = F.cross_entropy(logits, labels)
                grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
                grad_accum += grad
                
            # Average gradients
            grad_accum = grad_accum / self.N
            
            # Momentum Update (MI-FGSM style, standard for SSA)
            grad_norm = torch.mean(torch.abs(grad_accum), dim=(1, 2, 3), keepdim=True)
            grad_accum = grad_accum / (grad_norm + 1e-12)
            momentum = momentum + grad_accum
            
            # Apply perturbation
            x = x.detach() + self.alpha * momentum.sign()
            x = torch.clamp(x, x0 - self.eps, x0 + self.eps)
            x = torch.clamp(x, 0.0, 1.0)
            
        return x.detach()
