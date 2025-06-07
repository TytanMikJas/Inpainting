import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.baseline.encoder import Encoder
from src.models.ssl.MLP import MLP
from src.models.ssl.ViewGenerator import ViewGenerator
from copy import deepcopy


class BYOL(nn.Module):
    """
    BYOL (Bootstrap Your Own Latent) model for self-supervised learning.
    This implementation uses a convolutional encoder, a projector with BN, and a predictor with BN.
    Args:
        input_dim (int): Number of input channels (3 for RGB images).
        hidden_dim (int): Number of hidden channels in the encoder.
        num_residual_layers (int): Number of residual layers in the encoder.
        residual_hiddens (int): Number of hidden channels in each residual layer.
        device (str): Device to run the model on ('cpu' or 'cuda').
        image_size (int, optional): Size of the input images. Default is 64 for cats dataset.
        tau (float, optional): Update rate for the target network. Default is 0.99.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_residual_layers: int,
        residual_hiddens: int,
        mlp_hidden_dim: int = 1024,
        mask_size_ratio: float = 0.0,
        tau: float = 0.99,
        device: str = "cpu",
        image_size: int = 64,
    ):
        super().__init__()

        self.online_encoder = Encoder(
            in_channels=input_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
        )

        feature_map_size = image_size // 4
        self.flat_dim = hidden_dim * feature_map_size * feature_map_size

        self.online_projector = MLP(
            input_dim=self.flat_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_hidden_dim,
            plain_last=False,
        )
        self.online_predictor = MLP(
            input_dim=mlp_hidden_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_hidden_dim,
            plain_last=True,
        )

        self.target_encoder = self._copy_frozen_model(self.online_encoder)
        self.target_projector = self._copy_frozen_model(self.online_projector)

        self.device = device
        self.view_generator = ViewGenerator(
            mode='byol', device=self.device, image_size=image_size, mask_size_ratio=mask_size_ratio
        )
        self.tau = tau

    def forward(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v, v_prim = self.view_generator(x_batch)

        q1 = self.online_encoder(v)
        q1 = torch.flatten(q1, 1)
        q1 = self.online_predictor(self.online_projector(q1))

        q2 = self.online_encoder(v_prim)
        q2 = torch.flatten(q2, 1)
        q2 = self.online_predictor(self.online_projector(q2))

        with torch.no_grad():
            z1 = self.target_encoder(v_prim)
            z1 = torch.flatten(z1, 1)
            z1 = self.target_projector(z1)

            z2 = self.target_encoder(v)
            z2 = torch.flatten(z2, 1)
            z2 = self.target_projector(z2)

        q = torch.cat([q1, q2], dim=0)
        z = torch.cat([z1, z2], dim=0)

        return q, z
    
    def forward_repr(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)

    def byol_loss(self, q: torch.Tensor, z_prim: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=1)
        z_prim = F.normalize(z_prim, dim=1)
        return 2 - 2 * (q * z_prim).sum(dim=1).mean()

    @torch.no_grad()
    def update_target_network(self) -> None:
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data = (
                self.tau * target_param.data + (1 - self.tau) * online_param.data
            )

        for online_param, target_param in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_param.data = (
                self.tau * target_param.data + (1 - self.tau) * online_param.data
            )

    def _copy_frozen_model(self, online_model: nn.Module) -> nn.Module:
        target_model = deepcopy(online_model)
        for param in target_model.parameters():
            param.requires_grad = False
        return target_model
