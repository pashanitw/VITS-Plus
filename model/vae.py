import torch
from torch import nn, Tensor

from model.model import ModelConfig
from utils import sequence_mask
from typing import Tuple
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm
import torch.nn.functional as F


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
            dilation = dilation_rate ** i
            self.downs.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            Conv1d(
                                hidden_channels,
                                hidden_channels * 2,
                                kernel_size,
                                dilation=dilation,
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
                            nn.Conv1d(
                                hidden_channels,
                                hidden_channels * 2 if i < n_layers - 1 else hidden_channels,
                                kernel_size=1
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

                res_skip_acts = self.resblocks[i][i_down](acts)

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

        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.encoder = WavenetEncoder(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(self, y: torch.Tensor, y_lengths: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
              Shapes:
                  - x: :math:`[B, C, T]`
                  - x_lengths: :math:`[B, 1]`
                  - g: :math:`[B, C, 1]`
              """
        y_mask = sequence_mask(y_lengths)[:, None, :].to(y.dtype)

        y = self.pre(y) * y_mask
        # encode the input
        y = self.encoder(y, y_mask)

        # project encoded features
        stats = self.proj(y) * y_mask

        m, logs = torch.chunk(stats, 2, dim=1)

        z = m + torch.randn_like(m) * torch.exp(logs) * y_mask

        return z, m, logs, y_mask



def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


LRELU_SLOPE = 0.1




class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs_1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, dilation=dilation[i],
                              padding=get_padding(kernel_size, dilation=dilation[i]))
                ) for i in range(len(dilation))
            ]
        )

        self.convs_1.apply(init_weights)

        self.convs_2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, dilation=1,
                              padding=get_padding(kernel_size, dilation=1))
                ) for i in range(len(dilation))
            ]
        )

        self.convs_2.apply(init_weights)

    def forward(self, y: Tensor, y_mask=None) -> Tensor:
        for c1, c2 in zip(self.convs_1, self.convs_2):
            yt = F.leaky_relu(y, LRELU_SLOPE)

            if y_mask is not None:
                yt = yt * y_mask

            yt = c1(yt)
            yt = F.leaky_relu(yt, LRELU_SLOPE)

            if y_mask is not None:
                yt = yt * y_mask

            yt = c2(yt)

            y = yt + y

        if y_mask is not None:
            y = y * y_mask

        return y

    def remove_weight_norm(self):
        for l in self.convs_1:
            remove_weight_norm(l)

        for l in self.convs_2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: Tuple):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, dilation=dilation[i],
                              padding=get_padding(kernel_size, dilation=dilation[i]))
                ) for i in range(len(dilation))
            ]
        )

        self.convs.apply(init_weights)

    def forward(self, y: Tensor, y_mask=None) -> Tensor:
        for c in self.convs:
            yt = F.leaky_relu(y, LRELU_SLOPE)
            if y_mask is not None:
                yt = yt * y_mask
            yt = c(yt)
            y = y + yt

            if y_mask is not None:
                y = y * y_mask
            return y

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Decoder(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        self.num_kernels = len(args.resblock_kernel_sizes)
        self.num_upsamples = len(args.upsample_rates)
        self.conv_pre = nn.Conv1d(args.inter_channels, args.upsample_initial_channel, kernel_size=7, stride=1,
                                  padding=3)

        resblock = ResBlock1 if args.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(args.upsample_rates, args.upsample_kernel_sizes)):

            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        args.upsample_initial_channel //  (2 ** i),
                        args.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding = (k - u) // 2
                    )
                )
            )

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            ch = args.upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(
                nn.ModuleList(
                    [
                        resblock(
                            ch,
                            kernel_size=k,
                            dilation=d
                        )
                        for j, (k, d) in enumerate(zip(args.resblock_kernel_sizes, args.resblock_dilation_sizes))
                    ]
                )
            )

        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, stride=1, padding=3, bias=False)

        self.ups.apply(init_weights)

    def forward(self, y: Tensor) -> Tensor:
        y = self.conv_pre(y)

        for i in range(len(self.ups)):
            y = F.leaky_relu(y, LRELU_SLOPE)
            y = self.ups[i](y)

            ys = None

            for block in self.resblocks[i]:
                if ys is None:
                    ys = block(y)
                else:
                    ys = ys + block(y)

            y = ys / self.num_kernels

        y = F.leaky_relu(y)
        y = self.conv_post(y)
        y = torch.tanh(y)

        return y

