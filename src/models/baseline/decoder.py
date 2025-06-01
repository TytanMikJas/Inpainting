import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.baseline.Residual import ResidualStack


class Decoder(nn.Module):
    """
    Decoder module baseline for various VAE model implementations.
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of output channels for the first convolutional layer.
        num_residual_layers (int): Number of residual blocks in the stack.
        num_residual_hiddens (int): Number of output channels for the first convolutional layer in each block.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)
