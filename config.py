import torch
import os

class Config:
    def __init__(self):
        # Base directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # System Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = 10
        self.GPU_IDS = [0]
        self.DEBUG = 0
        self.VERBOSE = 1

        # Data Paths (Convert to Absolute Paths)
        self.TRAIN_IMAGES = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train"))
        self.TEST_IMAGES = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test"))
        self.VAL_IMAGES = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val"))
        self.MASKS_PATH = os.path.abspath(os.path.join(base_dir, "data_archive/Masks/train"))

        # Training Hyperparameters
        self.EPOCHS = 100
        self.EARLY_STOP_PATIENCE = 10
        self.BATCH_SIZE = 8
        self.NUM_WORKERS = 4
        self.IMAGE_SIZE = 256
        self.LEARNING_RATE = 0.0001
        self.D2G_LR_RATIO = 0.1
        self.BETA1 = 0.0
        self.BETA2 = 0.9
        self.WEIGHT_DECAY = 0.0001
        self.MAX_ITERS = 2000000

        # Loss Weights
        self.L1_LOSS_WEIGHT = 1.0
        self.FM_LOSS_WEIGHT = 10.0
        self.STYLE_LOSS_WEIGHT = 1.0
        self.CONTENT_LOSS_WEIGHT = 1.0
        self.ADV_LOSS_WEIGHT = 0.01

        # GAN Settings
        self.GAN_LOSS = "nsgan"
        self.GAN_POOL_SIZE = 0

        # Edge Detection
        self.EDGE_THRESHOLD = 0.5
        self.SIGMA = 2

# Initialize Config
config = Config()
