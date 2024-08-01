import torch
from torch import Tensor
from typing import List, Tuple, Union, Dict

mel_basis = {}
hann_window: Dict[str, Tensor] = {}


def spectrogram_torch(
    y: Tensor, n_fft: int, hop_size: int, win_size: int, center: bool = False
):
    """
    Compute the spectrogram of an audio signal using PyTorch.

    Args:
        y (torch.Tensor): The input audio signal.
        n_fft (int): The number of FFT components.
        hop_size (int): The number of samples between successive frames.
        win_size (int): The size of the window function.
        center (bool): Whether to pad the signal such that the t-th frame is centered at y[t * hop_size].

    Returns:
        torch.Tensor: The computed spectrogram.
    """
    # Check the range of the input audio signal

    if torch.min(y) < -1.0:
        print("Warning: min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("Warning: max value is ", torch.max(y))

    global hann_window

    # Generate a unique key for the Hann window based on its size, dtype, and device
    dtype_device = f"{y.dtype}_{y.device}"
    wnsize_dtype_device = f"{win_size}_{dtype_device}"

    # Check if the Hann window is already cached, if not, create and cache it
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # Pad the input audio signal to center the frames
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Compute the Short-Time Fourier Transform (STFT)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False
    )

    # Convert the complex-valued spectrogram to a magnitude spectrogram
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec
