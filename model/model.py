from dataclasses import dataclass
from typing import List
from spacy.symbols import nn
import torch
from torch import Tensor
import math

@dataclass
class ModelConfig:
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    upsample_rates: List[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: List[int]
    n_layers_q: int
    use_spectral_norm: bool
    n_speakers: int = 0
    gin_channels: int = 0
    use_sdp = True


class TextEncoder(nn.Module):
    def __init__(self, n_vocab:int,  args: ModelConfig):
        super().__init__()

        self.emb = nn.Embedding(n_vocab, args.hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, args.hidden_channels**-0.5)
        self.encoder = nn.LSTM()
        self.args = args

    def forward(self, x: Tensor, x_lengths: Tensor) -> Tensor:
        x = self.emb(x) * math.sqrt(self.args.hidden_channels)

class Generator(nn.Module):
    def __init__(self, n_vocab:int, args: ModelConfig, **kwargs):
        super().__init__()
