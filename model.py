from dataclasses import dataclass
from typing import List

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
