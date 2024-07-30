import os
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import csv
from torch import Tensor
from utils import cleaned_text_to_sequence,text_to_sequence


@dataclass
class DataConfig:
    training_files: str
    validation_files: str
    lang: str
    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: Optional[float]
    add_blank: bool
    n_speakers: int
    cleaned_text: bool = False
    min_text_len: int = 1
    max_text_len: int = 190


def get_data_path_list(file_path=None):
    files_list = []

    if file_path is not None:

        with open(file_path, "r", encoding="utf-8") as train_file:
            csv_reader = csv.reader(train_file, delimiter="|")
            for row in csv_reader:
                try:
                    audio_file_path = row[0]
                    transcription = row[1]
                    files_list.append((audio_file_path, transcription))
                except Exception as e:
                    print("======= skipping =========")

    return files_list


class AudioTextDataset(Dataset):
    """
    1) TextAudioLoader loads audio-text pairs,
    2) normalizes text, converts them to sequences of integers,
    3) computes spectrograms from audio files.
    """

    def __init__(self, data_path: str, config: DataConfig):

        random.seed(1234)
        self.audio_text_pairs = random.shuffle(get_data_path_list(data_path))
        self.config = config
        self._filter()

    def _filter(self):
        """
        Filters text and stores spectrogram lengths for bucketing.
        """
        new_audio_text_pairs = []
        spec_lengths = []
        for audio_file_path, transcription in self.audio_text_pairs:
            if (
                self.config.min_text_len
                <= len(transcription)
                <= self.config.max_text_len
            ):
                new_audio_text_pairs.append((audio_file_path, transcription))
                spec_lengths.append(
                    os.path.getsize(audio_file_path) // (2 * self.config.hop_length)
                )

        self.audio_text_pairs = new_audio_text_pairs
        self.spec_lengths = spec_lengths

    def get_audio_text_pairs(self, audiopath_and_text: Tuple[str, str]):
        """
        Separates the filename and text, retrieves the corresponding audio and spectrogram.
        """
        audio_path, text = audiopath_and_text
        text = self.get_text(text)
        spec, wav = self.get_audio(audio_path)

        return text, spec, wav

    def get_text(self, text: str):
        """
        Normalizes the text and converts it to a sequence of integers.
        """
        text_norm = None
        if self.config.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.config.lang)
        if self.config.add_blank:
            pass

        return torch.LongTensor(text_norm)
