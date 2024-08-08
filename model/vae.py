import torch
from torch import nn, Tensor
from model import ModelConfig
from utils import sequence_mask
from einops import rearrange
from typing import Tuple
class Wavenet(nn.Module):
    pass
class PosteriorEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            kernel_size: int,
            dilation_rate: int,
            n_layers: int,
            gin_channels: int = 0,
    ):

        """Posterior Encoder of VITS model.

                ::
                    x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

                Args:
                    in_channels : Number of input tensor channels.
                    out_channels : Number of output tensor channels.
                    hidden_channels : Number of hidden channels.
                    kernel_size : Kernel size of the WaveNet convolution layers.
                    dilation_rate : Dilation rate of the WaveNet layers.
                    num_layers : Number of the WaveNet layers.
                    cond_channels : Number of conditioning tensor channels. Defaults to 0.
                """

        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Linear(in_channels, hidden_channels)
        self.encoder = Wavenet(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Linear(hidden_channels, out_channels * 2)

    def forward(self, y: torch.Tensor, y_lengths:torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
              Shapes:
                  - x: :math:`[B, C, T]`
                  - x_lengths: :math:`[B, 1]`
                  - g: :math:`[B, C, 1]`
              """
        y_mask = sequence_mask(y_lengths)[:, None, :].to(y.dtype)

        y = self.pre(rearrange(y, 'b c t -> b t c'))
        y = rearrange(y, 'b t c -> b c t') * y_mask

        # encode the input
        y = self.encoder(y, y_mask)

        # project encoded features
        stats = self.proj(rearrange(y, 'b c t -> b t c'))
        stats = rearrange(stats, 'b t c -> b c t') * y_mask
        m, logs = torch.chunk(stats, 2, dim=1)

        z = m + torch.randn_like(m) * torch.exp(logs) * y_mask

        return z, m , logs, y_mask





