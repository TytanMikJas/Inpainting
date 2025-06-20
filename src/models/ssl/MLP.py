import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for BYOL projector and predictor.
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output features.
        plain_last (bool): If True, the last layer will not have BatchNorm and ReLU (for predictor). Default is False.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, plain_last: bool = False
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        if not plain_last:
            self.net.append(nn.BatchNorm1d(output_dim))
            self.net.append(nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
