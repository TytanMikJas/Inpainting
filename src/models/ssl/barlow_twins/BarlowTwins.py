import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.models.ssl.ViewGenerator import ViewGenerator
from src.models.baseline.encoder import Encoder
from src.models.ssl.MLP import MLP
from copy import deepcopy


class BarlowTwins(nn.Module):
    """
    Barlow Twins model for self-supervised learning.
    
    
    Args:
        input_dim (int): Number of input channels (3 for RGB images).
        hidden_dim (int): Number of hidden channels in the encoder.
        num_residual_layers (int): Number of residual layers in the encoder.
        residual_hiddens (int): Number of hidden channels in each residual layer.
        device (str): Device to run the model on ('cpu' or 'cuda').
        image_size (int, optional): Size of the input images. Default is 64 for cats dataset.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_residual_layers: int,
        residual_hiddens: int,
        mlp_hidden_dim: int = 1024,
        mask_size_ratio: float = 0.0,
        lambda_: float = 5e-3,
        device: str = "cpu",
        image_size: int = 64,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=input_dim,
            num_hiddens=hidden_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=residual_hiddens,
        )
            
        self.device = device
        self.view_generator = ViewGenerator(
            mode='barlow', device=self.device, image_size=image_size, mask_size_ratio=mask_size_ratio
        )
            
        self.lambda_ = lambda_
        
        feature_map_size = image_size // 4
        self.flat_dim = hidden_dim * feature_map_size * feature_map_size

        self.projector = MLP(
            input_dim=self.flat_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=mlp_hidden_dim,
            plain_last=False,
        )

    def forward(self, x_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            v, v_prim = self.view_generator(x_batch)
            
            feat1 = self.encoder(v)
            feat2 = self.encoder(v_prim)

            feat1 = torch.flatten(feat1, start_dim=1)
            feat2 = torch.flatten(feat2, start_dim=1)

            return self.projector(feat1), self.projector(feat2)
        
    def forward_repr(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)
    
    def barlow_twins_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z_a_norm = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        z_b_norm = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)

        batch_size = z_a.size(0)
        c = (z_a_norm.T @ z_b_norm) / batch_size  

        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()

        loss = on_diag + self.lambda_ * off_diag
        return loss
    
    def training_step(self, x_batch: torch.Tensor) -> torch.Tensor:
        z_a, z_b = self.forward(x_batch)
        return self.barlow_twins_loss(z_a=z_a, z_b=z_b)


    def _copy_frozen_model(self, online_model: nn.Module) -> nn.Module:
        target_model = deepcopy(online_model)
        for param in target_model.parameters():
            param.requires_grad = False
        return target_model


