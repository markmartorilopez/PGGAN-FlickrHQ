import torch
import torch.optim as optim
import config
from model import Generator
from utils import generate_examples, load_checkpoint

print(torch.cuda.is_available())

gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )

generate_examples(gen, 4)