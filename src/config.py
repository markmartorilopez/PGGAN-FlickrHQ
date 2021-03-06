import cv2
import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
DATASET = 'flickerHQ'
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 16, 16] # As paper
CHANNELS_IMG = 3
Z_DIM = 512  # As paper
IN_CHANNELS = 512  # As paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 5
