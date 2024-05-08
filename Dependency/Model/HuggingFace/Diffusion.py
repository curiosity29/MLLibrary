from diffusers import UNet2DModel

import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from functools import partial
from tqdm.auto import tqdm
tqdm = partial(tqdm, position=0, leave=True)

from pathlib import Path
import os, glob

from accelerate import notebook_launcher
from datetime import datetime

from dataclasses import dataclass
@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_training_steps = 200
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 2
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "Checkpoints"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "curiosity29/test_diffusion_24_4"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

def get_config():
    return TrainingConfig()



def get_model(image_size = 128, config = get_config()):
    model = UNet2DModel(
        sample_size= config.image_size,  # the target image resolution
        in_channels=4,  # the number of input channels, 3 for RGB images
        out_channels=4,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return model

def get_all(config = get_config()):
    model = get_model(config = config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.num_training_steps * config.num_epochs),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    return model, optimizer, lr_scheduler, noise_scheduler

def load_model(ckpt_path, config = get_config()):
    model = UNet2DModel.from_pretrained(ckpt_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.num_training_steps * config.num_epochs),
    )
    return dict(
        model = model,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
    )

from diffusers import DDPMScheduler
import matplotlib.pyplot as plt

def get_noise_scheduler():
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# sample_image = sample_image.to("cpu")
# noise = torch.randn(sample_image.shape)
# timesteps = torch.LongTensor([50])
# # noisy_image = noisy_image.to("cuda")
# noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

# Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
