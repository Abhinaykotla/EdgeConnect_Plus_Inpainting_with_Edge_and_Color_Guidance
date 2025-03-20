# Config.py - Configuration settings for EdgeConnect+ model training and evaluation
import torch
import os

class Config:
    def __init__(self):
        # Base directory where this script is located (for relative path resolution)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Data Paths (Convert to Absolute Paths)
        # Training data: ground truth (complete) images and input (masked) images
        self.TRAIN_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_gt"))
        self.TRAIN_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_input"))

        # Testing data: used for final model evaluation
        self.TEST_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_gt"))
        self.TEST_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_input"))

        # Validation data: used during training to monitor model performance
        self.VAL_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_gt"))
        self.VAL_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_input"))

        # Training Hyperparameters
        self.BATCH_SIZE = 12       # Number of images processed in each training iteration
        self.NUM_WORKERS = 6       # Number of parallel data loading workers
        self.PIN_MEMORY = True     # Speeds up CPU to GPU memory transfer when enabled
        self.EPOCHS = 100          # Maximum number of complete passes through the training dataset
        self.EARLY_STOP_PATIENCE = 5  # Stop training if no improvement after this many epochs
        self.IMAGE_SIZE = 256      # Size to resize all images (square dimensions)

        # Logging & Checkpoints
        self.VALIDATION_SAMPLE_EPOCHS = 5  # Run validation and save samples every N epochs
        self.TRAINING_SAMPLE_EPOCHS = 1    # Save training samples every N epochs
        self.MAX_BATCH_POINTS = 5000      # Max number of batch samples to save
        self.BATCH_SAMPLING_SIZE = 42      # Controls how often samples and logs are generated during training
        # Note: BATCH_SAMPLING_SIZE controls how often samples and logs are generated during training
        # For optimal visualization without gaps:
        # - Calculate total_batches = dataset_size / BATCH_SIZE
        #   (For CelebA: 162079 images / 12 batch size = 13506 batches)
        # - Choose BATCH_SAMPLING_SIZE as a divisor of total_batches
        #   (e.g., values like 13506/2, 13506/3, or 13506/4)
        # This ensures consistent sampling across the entire dataset

        # System Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
        
        # Output directories for model artifacts
        self.MODEL_CHECKPOINT_DIR = os.path.abspath(os.path.join(base_dir, "models/checkpoints"))  # Saved model states
        self.EPOCH_SAMPLES_DIR = os.path.abspath(os.path.join(base_dir, "models/generated_samples/epochs"))  # End-of-epoch samples
        self.BATCH_SAMPLES_DIR = os.path.abspath(os.path.join(base_dir, "models/generated_samples/batch"))    # Mid-epoch samples
        self.LOSS_PLOT_DIR = os.path.abspath(os.path.join(base_dir, "models/plots"))  # Loss visualization charts

        # Optimizer Parameters
        self.LEARNING_RATE = 0.00009       # Base learning rate for Adam optimizer Best value: 0.000493
        self.D2G_LR_RATIO = 0.0025          # Ratio between discriminator and generator learning rates
        self.BETA1 = 0.5                   # Adam optimizer beta1 parameter (momentum)
        self.BETA2 = 0.999                 # Adam optimizer beta2 parameter (RMSprop)
        self.WEIGHT_DECAY = 0.00005        # L2 regularization strength in Adam

        # Loss Weights (Controls the balance between different loss components)
        self.L1_LOSS_WEIGHT = 1            # Pixel-wise reconstruction loss weight
        self.ADV_LOSS_WEIGHT = 0.5         # Adversarial loss weight for generators
        self.FM_LOSS_WEIGHT = 1.5            # Feature matching loss weight
        self.STYLE_LOSS_WEIGHT = 250       # Style transfer loss weight
        self.CONTENT_LOSS_WEIGHT = 1.0     # Content preservation loss weight

        # Canny Edge Detection Parameters (For pre-processing input images)
        self.CANNY_THRESHOLD_LOW = 45      # Lower threshold for Canny edge detection sensitivity
        self.CANNY_THRESHOLD_HIGH = 140    # Upper threshold for Canny edge detection strength

        # GAN Settings
        self.GAN_LOSS = "nsgan"            # GAN loss function type (non-saturating GAN)
        self.ADV_LOSS_TYPE = "lsgan"       # Adversarial loss type (least squares GAN for stability)
        self.GAN_POOL_SIZE = 0             # Size of discriminator image buffer (0 = disabled)

        # Edge Detection Parameters (For model processing)
        self.EDGE_THRESHOLD = 0.5          # Threshold for edge map binarization
        self.SIGMA = 2                     # Gaussian blur sigma for edge smoothing

        # Training Control Parameters
        self.MAX_ITERS = 2000000           # Maximum number of iterations (backup to epoch limit)
        self.SEED = 42                     # Random seed for reproducibility
        self.GPU_IDS = [0]                 # GPU device IDs to use (for multi-GPU setups)
        self.DEBUG = 0                     # Debug level (0 = off, higher = more verbose)
        self.VERBOSE = 1                   # Verbosity level of training output

# Initialize Config
config = Config()