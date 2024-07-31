import os
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import csv
from torch import Tensor
from utils import (
    cleaned_text_to_sequence,
    text_to_sequence,
    load_wav_to_torch,
    intersperse,
)
from mel import spectrogram_torch


@dataclass
class DataConfig:
    training_files: str
    validation_files: str
    lang: str
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: Optional[float]
    add_blank: bool
    n_speakers: int
    spec_dir: str
    cleaned_text: bool = False
    max_wav_value: float = 32768.0
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

    def __getitem__(self, idx):
        return self.get_audio_text_pairs(self.audio_text_pairs[idx])

    def __len__(self):
        return len(self.audio_text_pairs)

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
        if self.config.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.config.lang)
        if self.config.add_blank:
            text_norm = intersperse(text_norm, 0)

        text_norm = torch.LongTensor(text_norm)

        return text_norm

    def get_audio(self, audio_path):
        audio, sr = load_wav_to_torch(audio_path)
        if sr != self.config.sampling_rate:
            raise ValueError(
                f"Sampling rate {sr} does not match {self.config.sampling_rate}"
            )

        audio_norm = audio / self.config.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = os.path.join(
            self.config.spec_dir,
            os.path.splitext(os.path.basename(audio_path))[0] + ".spec.pt",
        )

        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.config.filter_length,
                self.config.hop_length,
                self.config.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        return spec, audio_norm
