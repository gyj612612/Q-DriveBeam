from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureDenoiseAutoEncoder(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int = 128, use_vae: bool = False) -> None:
        super().__init__()
        self.use_vae = bool(use_vae)
        hidden_dim = max(feature_dim, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        enc_out = latent_dim * 2 if self.use_vae else latent_dim
        self.latent_head = nn.Linear(hidden_dim, enc_out)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Start conservative: mostly keep original feature at early training.
        self.mix_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        if self.use_vae:
            stats = self.latent_head(h)
            mu, logvar = stats.chunk(2, dim=-1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        else:
            z = self.latent_head(h)
            kl = x.new_tensor(0.0)

        recon = self.decoder(z)
        rec = F.mse_loss(recon, x)
        alpha = torch.sigmoid(self.mix_logit)
        denoised = (1.0 - alpha) * x + alpha * recon
        return denoised, rec, kl
