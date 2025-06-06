import torch
import torch.nn as nn
from typing import Tuple
from src.models.baseline.encoder import Encoder
from src.models.baseline.decoder import Decoder


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        residual_hiddens: int,
        num_residual_layers: int,
        latent_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 64,
    ):
        """
        Variational Autoencoder (VAE) for 64x64 RGB images.
        Args:
            input_dim (int): Number of input channels (3 for RGB).
            hidden_dim (int): Number of hidden channels in encoder/decoder.
            residual_hiddens (int): Hidden channels in residual blocks.
            num_residual_layers (int): Number of residual blocks.
            latent_dim (int): Size of the latent space.
        """
        super(VAE, self).__init__()
        self.device = device
        self.to(device)
        self.hidden_dim = hidden_dim

        self._input_shape = (input_dim, image_size, image_size)

        # Encoder
        self.encoder = Encoder(
            in_channels=input_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
        )

        # Calculate flattened feature map size from encoder output
        # 64x64 → 32x32 → 16x16 due to Conv2d stride=2 layers
        self._feature_map_size = image_size // 4
        self._flat_dim = hidden_dim * self._feature_map_size * self._feature_map_size

        # Latent Space
        self._fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self._fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Decoder
        self._fc_decode = nn.Linear(latent_dim, self._flat_dim)
        self._decoder = Decoder(
            in_channels=hidden_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
        )

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)

        return self._fc_mu(x), self._fc_logvar(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self._fc_decode(z)
        x = x.view(
            z.size(0), self.hidden_dim, self._feature_map_size, self._feature_map_size
        )
        x = self._decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        kl = self._kl_divergence(mu, logvar)

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        mean_kl_per_dim = kl_per_dim.mean(dim=0)
        num_active_dims = (mean_kl_per_dim > 0.01).sum().item()

        return {"recon": recon, "partial_loss": kl, "num_active_dims": num_active_dims}
