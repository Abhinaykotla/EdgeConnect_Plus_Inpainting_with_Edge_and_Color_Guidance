# config.py

import torch
import os

class Config_G1:
    def __init__(self):
        # Base directory inside Google Drive for Colab integration        
        base_drive_dir = "/content/drive/MyDrive/edgeconnect/"

        # Dataset paths configuration
        # Training data paths
        self.TRAIN_IMAGES_GT_G1 = "/content/CelebA/train_gt"      # Ground truth (complete) images
        self.TRAIN_IMAGES_INPUT_G1 = "/content/CelebA/train_input"  # Input images with masks

        # Testing data paths
        self.TEST_IMAGES_GT_G1 = os.path.join(base_drive_dir, "data_archive/CelebA/test_gt")      # Ground truth test images
        self.TEST_IMAGES_INPUT_G1 = os.path.join(base_drive_dir, "data_archive/CelebA/test_input")  # Input test images

        # Validation data paths
        self.VAL_IMAGES_GT_G1 = os.path.join(base_drive_dir, "data_archive/CelebA/val_gt")      # Ground truth validation images
        self.VAL_IMAGES_INPUT_G1 = os.path.join(base_drive_dir, "data_archive/CelebA/val_input")  # Input validation images

        # Training Hyperparameters
        self.BATCH_SIZE_G1 = 192        # Number of images processed in each training iteration
        self.NUM_WORKERS_G1 = 12        # Number of parallel data loading workers
        self.PIN_MEMORY_G1 = True       # Speeds up CPU to GPU memory transfer when enabled
        self.EPOCHS_G1 = 250            # Total number of training epochs to run
        self.EARLY_STOP_PATIENCE_G1 = 5 # Stop training if no improvement after this many epochs
        self.IMAGE_SIZE_G1 = 256        # Size to resize all images (square dimensions)

        # Logging & Checkpoints
        self.VALIDATION_SAMPLE_EPOCHS_G1 = 5  # Run validation every N epochs
        self.TRAINING_SAMPLE_EPOCHS_G1 = 1    # Save training samples every N epochs
        self.MAX_BATCH_POINTS_G1 = 10000       # Max number of batch samples to save
        self.BATCH_SAMPLING_SIZE_G1 = 169     # Controls how often samples and logs are generated during training
        # Note: BATCH_SAMPLING_SIZE controls how often samples and logs are generated during training
        # For optimal visualization without gaps:
        # - Calculate total_batches = dataset_size / BATCH_SIZE
        #   (For CelebA: 162079 images / 192 batch size = 845 batches)
        # - Choose BATCH_SAMPLING_SIZE as a divisor of total_batches
        #   (e.g., values like 845/5 = 169)
        # This ensures consistent sampling across the entire dataset

        # System Settings
        self.DEVICE_G1 = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
        
        # Output directories for model artifacts
        self.MODEL_CHECKPOINT_DIR_G1 = os.path.join(base_drive_dir, "models/checkpoints")  # For saved model states
        self.EPOCH_SAMPLES_DIR_G1 = os.path.join(base_drive_dir, "models/generated_samples/epochs")  # For epoch-based samples
        self.BATCH_SAMPLES_DIR_G1 = os.path.join(base_drive_dir, "models/generated_samples/batch")  # For batch-based samples
        self.LOSS_PLOT_DIR_G1 = os.path.join(base_drive_dir, "models/plots")  # For loss visualizations

        # Optimizer Parameters
        self.LEARNING_RATE_G1= 0.00009       # Base learning rate for Adam optimizer Best : 0.000493
        self.D2G_LR_RATIO_G1 = 0.0025          # Ratio between discriminator and generator learning rates
        self.BETA1_G1 = 0.5                   # Adam optimizer beta1 parameter (momentum)
        self.BETA2_G1 = 0.999                 # Adam optimizer beta2 parameter (RMSprop)
        self.WEIGHT_DECAY_G1 = 0.00005        # L2 regularization strength in Adam

        # Loss Weights (Controls the balance between different loss components)
        self.L1_LOSS_WEIGHT_G1 = 1            # Pixel-wise reconstruction loss weight
        self.ADV_LOSS_WEIGHT_G1 = 0.5         # Adversarial loss weight for generators
        self.FM_LOSS_WEIGHT_G1 = 1.5            # Feature matching loss weight
        self.STYLE_LOSS_WEIGHT_G1 = 250       # Style transfer loss weight
        self.CONTENT_LOSS_WEIGHT_G1 = 1.0     # Content preservation loss weight

        # Canny Edge Detection Parameters (For pre-processing)
        self.CANNY_THRESHOLD_LOW_G1 = 45      # Lower threshold for Canny edge detection
        self.CANNY_THRESHOLD_HIGH_G1 = 140    # Upper threshold for Canny edge detection

        # GAN Settings
        self.GAN_LOSS_G1 = "nsgan"           # Type of GAN loss function (non-saturating GAN)
        self.ADV_LOSS_TYPE_G1 = "lsgan"      # Adversarial loss type (least squares GAN)
        self.GAN_POOL_SIZE_G1 = 0            # Size of discriminator image buffer (0 = no buffer)

        # Edge Detection Parameters
        self.EDGE_THRESHOLD_G1 = 0.5         # Threshold for edge map binarization
        self.SIGMA_G1 = 2                    # Gaussian blur sigma for edge smoothing

        # Training Control Parameters
        self.MAX_ITERS_G1 = 2000000          # Maximum number of iterations (backup to epoch limit)
        self.SEED_G1 = 42                    # Random seed for reproducibility
        self.GPU_IDS_G1 = [0]                # GPU device IDs to use (for multi-GPU setups)
        self.DEBUG_G1 = 0                    # Debug level (0 = off, higher = more verbose)
        self.VERBOSE_G1 = 1                  # Verbosity level of output (0 = minimal, 1 = normal)

# Initialize Config
config = Config_G1()