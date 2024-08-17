import torch
from torch import nn, Tensor
from utils import sequence_mask
from einops import rearrange
from typing import Tuple
from torch.nn import Conv1d
from torch.nn.utils import weight_norm


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

def get_padding(kernel_size, dilation):
    return int((kernel_size * dilation - dilation) / 2)

class WavenetEncoder(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.downs = nn.ModuleList()
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.drop = nn.Dropout(p_dropout)


        for i in range(n_layers):
            dilation = dilation_rate**i
            self.downs.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            Conv1d(
                                hidden_channels,
                                hidden_channels * 2,
                                kernel_size,
                                dilation= dilation,
                                padding=get_padding(kernel_size, dilation=dilation)
                            )
                        )
                    ]
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.downs)):
            self.resblocks.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            nn.Linear(
                                hidden_channels,
                                hidden_channels * 2 if i < n_layers - 1 else hidden_channels
                            )
                        )
                    ]
                )
            )

    def forward(self, y: Tensor, y_mask: Tensor):

        output = torch.zeros_like(y)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        for i in range(self.n_layers):
            for i_down in range(len(self.downs[i])):
                y_down = self.downs[i][i_down](y)

                g_l = torch.zeros_like(y_down)

                acts = fused_add_tanh_sigmoid_multiply(y_down, g_l, n_channels_tensor)

                acts = self.drop(acts)

                res_skip_acts = self.resblocks[i][i_down](rearrange(acts, 'b c t -> b t c'))
                res_skip_acts = rearrange(res_skip_acts, 'b t c -> b c t')

                if i < self.n_layers - 1:
                    res_acts = res_skip_acts[:, :self.hidden_channels, :]
                    y = (y + res_acts) * y_mask
                    output = output + res_skip_acts[:, self.hidden_channels:, :]
                else:
                    output = output + res_skip_acts

        return output * y_mask






class PosteriorEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_channels: int,
            n_layers: int,
            kernel_size: int = 5,
            dilation_rate: int = 1,
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
        self.encoder = WavenetEncoder(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
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





class Decoder(nn.Module):
    def __init__(
            self,
    ):
        pass