from scipy.io.wavfile import read
import torch
from lang_cleaner.eng import english_cleaners, get_english_symbols
import numpy as np
import logging
import sys
from logging.handlers import RotatingFileHandler
from lang_cleaner.eng import get_english_symbols


def cleaned_text_to_sequence(cleaned_text: str, lang: str):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if lang == "en":
        _, _symbol_to_id, _ = get_english_symbols()
    else:
        raise ValueError(f"Language {lang} is not supported.")

    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def text_to_sequence(text: str, lang: str):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if lang == "en":
        cleaner_fn = english_cleaners
        _, _symbol_to_id, _ = get_english_symbols()

    else:
        raise ValueError(f"Language {lang} not supported")
    sequence = []
    if cleaner_fn is not None:
        text = cleaner_fn(text)
    for symbol in text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def sequence_mask(x_lengths: torch.Tensor):

    max_len = x_lengths.max()

    mask = torch.arange(max_len, dtype=x_lengths.dtype, device=x_lengths.device)

    return mask[None, :] < x_lengths[:, None]


def create_attn_mask(x_lengths: torch.Tensor):
    max_len = x_lengths.max()
    batch_size = x_lengths.size(0)

    # Create a mask of shape [B, max_len]
    mask = torch.arange(max_len, device=x_lengths.device).expand(batch_size, max_len)
    mask = mask < x_lengths.unsqueeze(1)

    # Convert to float
    mask = mask.float()

    # Transform the mask
    # Use torch.finfo(mask.dtype).min instead of float("-inf") for better numerical stability
    transformed_mask = (1.0 - mask) * torch.finfo(mask.dtype).min

    # The mask should be broadcastable to [B, 1, max_len, max_len]
    # for compatibility with scaled_dot_product_attention
    transformed_mask = transformed_mask.unsqueeze(1).unsqueeze(1)

    return transformed_mask


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger that logs to both file and console"""

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def get_vocab_len(lang: str):
    if lang == "en":
        symbols, _, _ = get_english_symbols()
        return len(symbols)
    else:
        raise ValueError(f"Language {lang} is not supported.")
