# config.py

import torch
import os

class Config:
    def __init__(self):
        # Base directory inside Google Drive
        base_drive_dir = "/content/drive/MyDrive/edgeconnect/"

        # Dataset paths in Drive
        self.TRAIN_IMAGES_GT = "content/CelebA/train_gt"
        self.TRAIN_IMAGES_INPUT = "content/CelebA/train_input"

        self.TEST_IMAGES_GT = os.path.join(base_drive_dir, "data_archive/CelebA/test_gt")
        self.TEST_IMAGES_INPUT = os.path.join(base_drive_dir, "data_archive/CelebA/test_input")

        self.VAL_IMAGES_GT = os.path.join(base_drive_dir, "data_archive/CelebA/val_gt")
        self.VAL_IMAGES_INPUT = os.path.join(base_drive_dir, "data_archive/CelebA/val_input")

        # Training Hyperparameters
        self.BATCH_SIZE = 12
        self.NUM_WORKERS = 6
        self.EPOCHS = 100
        self.EARLY_STOP_PATIENCE = 5  # Updated for faster tracking
        self.IMAGE_SIZE = 256

        # Logging & Checkpoints
        self.VALIDATION_SAMPLE_EPOCHS = 5  # Updated for faster tracking
        self.TRAINING_SAMPLE_EPOCHS = 1  # Updated for faster tracking
        self.BATCH_SAMPLING_SIZE = 50  # Updated for faster/slower tracking

        # System Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL_CHECKPOINT_DIR = os.path.join(base_drive_dir, "models/checkpoints")
        self.EPOCH_SAMPLES_DIR = os.path.join(base_drive_dir, "moedels/generated_samples/epochs")
        self.BATCH_SAMPLES_DIR = os.path.join(base_drive_dir, "models/generated_samples/batch")
        self.LOSS_PLOT_DIR = os.path.join(base_drive_dir, "models/plots")

        self.LEARNING_RATE = 0.0001
        self.D2G_LR_RATIO = 0.02  # Reduced to slow down D1
        self.BETA1 = 0.5
        self.BETA2 = 0.999
        self.WEIGHT_DECAY = 0.00005  # Reduced for stability

        # Loss Weights (Optimized)
        self.L1_LOSS_WEIGHT = 1
        self.ADV_LOSS_WEIGHT = 1
        self.FM_LOSS_WEIGHT = 10
        self.STYLE_LOSS_WEIGHT = 250
        self.CONTENT_LOSS_WEIGHT = 1.0

        # Canny Edge Detection Parameters
        self.CANNY_THRESHOLD_LOW = 60
        self.CANNY_THRESHOLD_HIGH = 160

        # GAN Settings
        self.GAN_LOSS = "nsgan"
        self.ADV_LOSS_TYPE = "lsgan"
        self.GAN_POOL_SIZE = 0

        # Edge Detection Parameters
        self.EDGE_THRESHOLD = 0.5
        self.SIGMA = 2

        self.MAX_ITERS = 2000000
        self.SEED = 42
        self.GPU_IDS = [0]
        self.DEBUG = 0
        self.VERBOSE = 1

# Initialize Config
config = Config()
