# config.py

import torch
import os

class Config:
    def __init__(self):
        # Base directory inside Google Drive for Colab integration
        base_drive_dir = "/content/drive/MyDrive/edgeconnect/"

        # Dataset paths configuration
        # Training data paths
        self.TRAIN_IMAGES_GT = "/content/CelebA/train_gt"      # Ground truth (complete) images
        self.TRAIN_IMAGES_INPUT = "/content/CelebA/train_input"  # Input images with masks

        # Testing data paths
        self.TEST_IMAGES_GT = os.path.join(base_drive_dir, "data_archive/CelebA/test_gt")      # Ground truth test images
        self.TEST_IMAGES_INPUT = os.path.join(base_drive_dir, "data_archive/CelebA/test_input")  # Input test images

        # Validation data paths
        self.VAL_IMAGES_GT = os.path.join(base_drive_dir, "data_archive/CelebA/val_gt")      # Ground truth validation images
        self.VAL_IMAGES_INPUT = os.path.join(base_drive_dir, "data_archive/CelebA/val_input")  # Input validation images

        # Training Hyperparameters
        self.BATCH_SIZE = 192        # Number of images processed in each training iteration
        self.NUM_WORKERS = 12        # Number of parallel data loading workers
        self.PIN_MEMORY = True       # Speeds up CPU to GPU memory transfer when enabled
        self.EPOCHS = 250            # Total number of training epochs to run
        self.EARLY_STOP_PATIENCE = 10 # Stop training if no improvement after this many epochs
        self.IMAGE_SIZE = 256        # Size to resize all images (square dimensions)

        # Logging & Checkpoints
        self.VALIDATION_SAMPLE_EPOCHS = 5  # Run validation every N epochs
        self.TRAINING_SAMPLE_EPOCHS = 1    # Save training samples every N epochs
        self.MAX_BATCH_POINTS = 10000       # Max number of batch samples to save
        self.BATCH_SAMPLING_SIZE = 169     # Controls how often samples and logs are generated during training
        # Note: BATCH_SAMPLING_SIZE controls how often samples and logs are generated during training
        # For optimal visualization without gaps:
        # - Calculate total_batches = dataset_size / BATCH_SIZE
        #   (For CelebA: 162079 images / 192 batch size = 845 batches)
        # - Choose BATCH_SAMPLING_SIZE as a divisor of total_batches
        #   (e.g., values like 845/5 = 169)
        # This ensures consistent sampling across the entire dataset

        # System Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
        
        # Output directories for model artifacts
        self.MODEL_CHECKPOINT_DIR = os.path.join(base_drive_dir, "models/checkpoints")  # For saved model states
        self.EPOCH_SAMPLES_DIR = os.path.join(base_drive_dir, "models/generated_samples/epochs")  # For epoch-based samples
        self.BATCH_SAMPLES_DIR = os.path.join(base_drive_dir, "models/generated_samples/batch")  # For batch-based samples
        self.LOSS_PLOT_DIR = os.path.join(base_drive_dir, "models/plots")  # For loss visualizations

        # Optimizer Parameters
        self.LEARNING_RATE = 0.0001       # Base learning rate for Adam optimizer
        self.D2G_LR_RATIO = 0.1           # Ratio between discriminator and generator learning rates
        self.BETA1 = 0.0                  # Adam optimizer beta1 parameter (momentum)
        self.BETA2 = 0.9                  # Adam optimizer beta2 parameter (RMSprop)
        self.WEIGHT_DECAY = 0.00005        # L2 regularization strength in Adam

        # Loss Weights (Controls the balance between different loss components)
        self.L1_LOSS_WEIGHT = 0.7            # Pixel-wise reconstruction loss weight
        self.ADV_LOSS_WEIGHT = 1.5         # Adversarial loss weight for generators
        self.FM_LOSS_WEIGHT = 5.0            # Feature matching loss weight
        self.STYLE_LOSS_WEIGHT = 250       # Style transfer loss weight
        self.CONTENT_LOSS_WEIGHT = 1.0     # Content preservation loss weight

        # Canny Edge Detection Parameters (For pre-processing)
        self.CANNY_THRESHOLD_LOW = 45      # Lower threshold for Canny edge detection
        self.CANNY_THRESHOLD_HIGH = 140    # Upper threshold for Canny edge detection

        # GAN Settings
        self.GAN_LOSS = "nsgan"           # Type of GAN loss function (non-saturating GAN)
        self.ADV_LOSS_TYPE = "lsgan"      # Adversarial loss type (least squares GAN)
        self.GAN_POOL_SIZE = 0            # Size of discriminator image buffer (0 = no buffer)

        # Edge Detection Parameters
        self.EDGE_THRESHOLD = 0.5         # Threshold for edge map binarization
        self.SIGMA = 2                    # Gaussian blur sigma for edge smoothing

        # Training Control Parameters
        self.MAX_ITERS = 2000000          # Maximum number of iterations (backup to epoch limit)
        self.SEED = 42                    # Random seed for reproducibility
        self.GPU_IDS = [0]                # GPU device IDs to use (for multi-GPU setups)
        self.DEBUG = 0                    # Debug level (0 = off, higher = more verbose)
        self.VERBOSE = 1                  # Verbosity level of output (0 = minimal, 1 = normal)

# Initialize Config
config = Config()