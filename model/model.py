from dataclasses import dataclass
from typing import List
import torch
from torch import Tensor, nn
import math
from .llama import llama_encoder, LLM_Args
from utils import create_attn_mask
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

        llm_args = LLM_Args(
            num_layers=args.n_layers,
            num_heads=args.n_heads,
            attn_dropout=args.p_dropout,
            embed_dim=args.hidden_channels,
            vocab_size=n_vocab,
            max_seq_len=512,
            norm_eps=1e-5,
            rope_base=10000
        )

        self.encoder = llama_encoder(llm_args)

        self.output_proj = nn.Linear(args.hidden_channels, args.hidden_channels * 2, bias=False)
    def forward(self, x: Tensor, x_lengths: Tensor):
        attn_mask = create_attn_mask(x_lengths)
        x = self.encoder(x, attn_mask)
        stats = self.output_proj(x)
        m, logs = stats.chunk(2, dim=-1)

        return x, m, logs, attn_mask
class Generator(nn.Module):
    def __init__(self, n_vocab:int, args: ModelConfig, **kwargs):
        super().__init__()
        self.text_encoder = TextEncoder(n_vocab, args)

    def forward(self, x: Tensor, x_lengths: Tensor):
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.test = nn.Linear(64, 64)

    def forward(self, x: Tensor, x_lengths: Tensor):
        return x