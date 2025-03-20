# Config.py - Configuration settings for EdgeConnect+ model training and evaluation
import torch
import os

class Config_G1:
    def __init__(self):
        # Base directory where this script is located (for relative path resolution)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Data Paths (Convert to Absolute Paths)
        # Training data: ground truth (complete) images and input (masked) images
        self.TRAIN_IMAGES_GT_G1 = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_gt"))
        self.TRAIN_IMAGES_INPUT_G1 = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_input"))

        # Testing data: used for final model evaluation
        self.TEST_IMAGES_GT_G1 = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_gt"))
        self.TEST_IMAGES_INPUT_G1 = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_input"))

        # Validation data: used during training to monitor model performance
        self.VAL_IMAGES_GT_G1 = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_gt"))
        self.VAL_IMAGES_INPUT_G1 = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_input"))

        # Training Hyperparameters
        self.BATCH_SIZE_G1 = 12       # Number of images processed in each training iteration
        self.NUM_WORKERS_G1 = 6       # Number of parallel data loading workers
        self.PIN_MEMORY_G1 = True     # Speeds up CPU to GPU memory transfer when enabled
        self.EPOCHS_G1 = 100          # Maximum number of complete passes through the training dataset
        self.EARLY_STOP_PATIENCE_G1 = 5  # Stop training if no improvement after this many epochs
        self.IMAGE_SIZE_G1 = 256      # Size to resize all images (square dimensions)

        # Logging & Checkpoints
        self.VALIDATION_SAMPLE_EPOCHS_G1 = 5  # Run validation and save samples every N epochs
        self.TRAINING_SAMPLE_EPOCHS_G1 = 1    # Save training samples every N epochs
        self.BATCH_SAMPLING_SIZE_G1 = 42      # Controls how often samples and logs are generated during training
        # Note: BATCH_SAMPLING_SIZE_G1 controls how often samples and logs are generated during training
        # For optimal visualization without gaps:
        # - Calculate total_batches = dataset_size / BATCH_SIZE_G1
        #   (For CelebA: 162079 images / 12 batch size = 13506 batches)
        # - Choose BATCH_SAMPLING_SIZE_G1 as a divisor of total_batches
        #   (e.g., values like 13506/2, 13506/3, or 13506/4)
        # This ensures consistent sampling across the entire dataset

        # System Settings
        self.DEVICE_G1 = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
        
        # Output directories for model artifacts
        self.MODEL_CHECKPOINT_DIR_G1 = os.path.abspath(os.path.join(base_dir, "models/checkpoints"))  # Saved model states
        self.EPOCH_SAMPLES_DIR_G1 = os.path.abspath(os.path.join(base_dir, "models/generated_samples/epochs"))  # End-of-epoch samples
        self.BATCH_SAMPLES_DIR_G1 = os.path.abspath(os.path.join(base_dir, "models/generated_samples/batch"))    # Mid-epoch samples
        self.LOSS_PLOT_DIR_G1 = os.path.abspath(os.path.join(base_dir, "models/plots"))  # Loss visualization charts

        # Optimizer Parameters
        self.LEARNING_RATE_G1 = 0.0001        # Base learning rate for Adam optimizer
        self.D2G_LR_RATIO_G1 = 0.02           # Discriminator learning rate ratio (relative to generator)
        self.BETA1_G1 = 0.5                   # Adam optimizer beta1 parameter (momentum)
        self.BETA2_G1 = 0.999                 # Adam optimizer beta2 parameter (RMSprop)
        self.WEIGHT_DECAY_G1 = 0.00005        # L2 regularization strength in Adam optimizer

        # Loss Weights (Controls the balance between different loss components)
        self.L1_LOSS_WEIGHT_G1 = 1            # Pixel-wise reconstruction loss weight
        self.ADV_LOSS_WEIGHT_G1 = 1           # Adversarial loss weight for generators
        self.FM_LOSS_WEIGHT_G1 = 5            # Feature matching loss weight (for better feature preservation)
        self.STYLE_LOSS_WEIGHT_G1 = 250       # Style transfer loss weight (for texture consistency)
        self.CONTENT_LOSS_WEIGHT_G1 = 1.0     # Content preservation loss weight (for semantic consistency)

        # Canny Edge Detection Parameters (For pre-processing input images)
        self.CANNY_THRESHOLD_LOW_G1 = 45      # Lower threshold for Canny edge detection sensitivity
        self.CANNY_THRESHOLD_HIGH_G1 = 140    # Upper threshold for Canny edge detection strength

        # GAN Settings
        self.GAN_LOSS_G1 = "nsgan"            # GAN loss function type (non-saturating GAN)
        self.ADV_LOSS_TYPE_G1 = "lsgan"       # Adversarial loss type (least squares GAN for stability)
        self.GAN_POOL_SIZE_G1 = 0             # Size of discriminator image buffer (0 = disabled)

        # Edge Detection Parameters (For model processing)
        self.EDGE_THRESHOLD_G1 = 0.5          # Threshold for edge map binarization
        self.SIGMA_G1 = 2                     # Gaussian blur sigma for edge smoothing

        # Training Control Parameters
        self.MAX_ITERS_G1 = 2000000           # Maximum number of iterations (backup to epoch limit)
        self.SEED_G1 = 42                     # Random seed for reproducibility
        self.GPU_IDS_G1 = [0]                 # GPU device IDs to use (for multi-GPU setups)
        self.DEBUG_G1 = 0                     # Debug level (0 = off, higher = more verbose)
        self.VERBOSE_G1 = 1                   # Verbosity level of training output

# Initialize Config
config_g1 = Config_G1()