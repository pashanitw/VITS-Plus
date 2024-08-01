import torch
from torch.utils.data import DataLoader
from sconf import Config
from dataset import AudioTextDataset, text_audio_collate_fn, DataConfig
import munch

config_file = "./config.yaml"

config = Config(config_file)
data_dict = config.data.toDict()
data_config  = DataConfig(**data_dict)

print(config.model)

train_Dataset = AudioTextDataset(config.data.training_files, data_config)

train_loader = DataLoader(train_Dataset, batch_size=config.train.batch_size, collate_fn=text_audio_collate_fn, shuffle=True)

for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
    print(batch_idx)
    break