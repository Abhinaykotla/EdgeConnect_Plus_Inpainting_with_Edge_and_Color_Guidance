import torch
import os

class Config:
    def __init__(self):
        # Base directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Data Paths (Convert to Absolute Paths)
        self.TRAIN_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_gt"))
        self.TRAIN_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_input"))

        self.TEST_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_gt"))
        self.TEST_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_input"))

        self.VAL_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_gt"))
        self.VAL_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_input"))

        # Training Hyperparameters
        self.BATCH_SIZE = 4  # Adjusted for better GPU usage
        self.NUM_WORKERS = 4  # More workers for faster data loading
        self.EPOCHS = 100  # Updated to reflect full training cycle
        self.EARLY_STOP_PATIENCE = 10  # Stop training if no improvement
        self.IMAGE_SIZE = 256  # Ensure consistency with dataset
        self.LEARNING_RATE = 0.0001  # Updated from hardcoded value in train.py
        self.D2G_LR_RATIO = 0.1  # Discriminator learning rate relative to G1
        self.BETA1 = 0.5  # Stability in Adam optimizer
        self.BETA2 = 0.999  # Adjusted momentum
        self.WEIGHT_DECAY = 0.0001  # Regularization for training
        self.MAX_ITERS = 2000000  # Maximum iterations if needed

        # System Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = 42  # Random seed for reproducibility
        self.GPU_IDS = [0]  # Select GPU for training
        self.DEBUG = 0  # Debug mode off by default
        self.VERBOSE = 1  # Print detailed training logs

        # Loss Weights (Now used in train.py)
        self.L1_LOSS_WEIGHT = 5.0  # Adjusted for smooth edges
        self.ADV_LOSS_WEIGHT = 0.5  # Ensures adversarial balance
        self.FM_LOSS_WEIGHT = 10.0  # Feature matching loss importance
        self.STYLE_LOSS_WEIGHT = 1.0  # Preserves texture details
        self.CONTENT_LOSS_WEIGHT = 1.0  # Keeps content fidelity

        # GAN Settings
        self.GAN_LOSS = "nsgan"  # Non-saturating GAN loss
        self.GAN_POOL_SIZE = 0  # No fake image pooling

        # Edge Detection Parameters
        self.EDGE_THRESHOLD = 0.5  # Controls edge sharpness
        self.SIGMA = 2  # Smoothing for edge detection

        # Logging & Checkpoints
        self.SAVE_IMAGES_EVERY = 5  # Save training images every N epochs
        self.SAVE_MODEL_EVERY = 10  # Save model checkpoints every N epochs
        self.OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, "output"))  # Save generated images here

# Initialize Config
config = Config()
