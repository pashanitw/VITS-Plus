import torch
import os
from torch.utils.data import DataLoader
from sconf import Config
from dataset import AudioTextDataset, text_audio_collate_fn, DataConfig
from model.model import Generator, MultiPeriodDiscriminator
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from utils import setup_logger, get_vocab_len

logger = setup_logger("model_training", "train.log")


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

config_file = "./config.yaml"

config = Config(config_file)
data_dict = config.data.toDict()
data_config = DataConfig(**data_dict)

print(config.model)

accelerator = Accelerator()

train_Dataset = AudioTextDataset(config.data.training_files, data_config)


def train_and_evaluate(loaders, nets):
    train_loader, _ = loaders
    net_g, net_d = nets
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(
        train_loader
    ):
        net_g(x, x_lengths, y, y_lengths)


if accelerator.is_main_process:
    logger.info(config)

torch.manual_seed(config.train.seed)


train_Dataset = AudioTextDataset(config.data.training_files, data_config)

train_loader = DataLoader(
    train_Dataset,
    batch_size=config.train.batch_size,
    collate_fn=text_audio_collate_fn,
    shuffle=False,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=16,
    num_workers=config.train.num_workers,
)


net_g = Generator(get_vocab_len(config.data.lang), config.model)

net_d = MultiPeriodDiscriminator()
for name, param in net_g.named_parameters():
    if not param.requires_grad:
        print(name, "not requires_grad")


optim_g = torch.optim.AdamW(
    net_g.parameters(),
    config.train.learning_rate,
    betas=config.train.betas,
    eps=config.train.eps,
)
optim_d = torch.optim.AdamW(
    net_d.parameters(),
    config.train.learning_rate,
    betas=config.train.betas,
    eps=config.train.eps,
)


world_size = accelerator.num_processes
num_epochs = config.train.epochs
one_epoch_steps = len(train_loader) // world_size
total_steps = num_epochs * one_epoch_steps

print("====> Total steps: {}".format(total_steps))
print("====> one epoch steps: {}".format(one_epoch_steps))
warmup_steps = int(0.1 * total_steps)
print("====> warmup steps: {}".format(warmup_steps))


scheduler_g = get_cosine_schedule_with_warmup(
    optim_g, num_warmup_steps=5000, num_training_steps=total_steps
)


scheduler_d = get_cosine_schedule_with_warmup(
    optim_d, num_warmup_steps=5000, num_training_steps=total_steps
)

net_g, net_d, train_loader, optim_g, optim_d, scheduler_g, scheduler_d = (
    accelerator.prepare(
        net_g, net_d, train_loader, optim_g, optim_d, scheduler_g, scheduler_d
    )
)

if config.train.ckpt_path != "":

    accelerator.load_state(config.train.ckpt_path)

    global_step = int(os.path.basename(config.train.ckpt_path).split("_")[-1])

    epoch_str = global_step // len(train_loader)

    # epoch_str = epoch_str + 1 #temp change

    global_step = epoch_str * len(train_loader)

    # Option 2: If your scheduler doesn't have 'base_lrs', you might need to set the learning rate in the optimizer
    # fixed_lr = 3e-5
    # for param_group in optim_g.param_groups:
    #     param_group['lr'] = fixed_lr
    #
    # for param_group in optim_d.param_groups:
    #     param_group['lr'] = fixed_lr

else:
    global_step = 0
    epoch_str = 0

print("=== current global step is ===", global_step)

# for _ in range(global_step):
#     scheduler_g.step()
#     scheduler_d.step()


rank = accelerator.process_index
for epoch in range(epoch_str, config.train.epochs + 1):
    if accelerator.is_main_process:
        train_and_evaluate([train_loader, None], [net_g, None])
    else:
        train_and_evaluate([train_loader, None], [net_g, net_d])

# wandb.finish()
