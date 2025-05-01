# Config.py - Configuration settings for EdgeConnect+ model training and evaluation
import torch
import os

class Config:
    def __init__(self):
        # Base directory where this script is located (for relative path resolution)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        #######################################################################
        # GENERAL CONFIGURATIONS (Shared between G1 and G2)
        #######################################################################
        
        # System Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
        
        # Training Hyperparameters
        self.BATCH_SIZE = 8              # Number of images processed in each training iteration
        self.NUM_WORKERS = 4              # Number of parallel data loading workers
        self.PIN_MEMORY = True            # Speeds up CPU to GPU memory transfer when enabled
        self.EPOCHS = 100                 # Maximum number of complete passes through the training dataset
        self.EARLY_STOP_PATIENCE = 5      # Stop training if no improvement after this many epochs
        self.IMAGE_SIZE = 256             # Size to resize all images (square dimensions)
        
        # Common Logging Parameters
        self.VALIDATION_SAMPLE_EPOCHS = 5  # Run validation and save samples every N epochs
        self.TRAINING_SAMPLE_EPOCHS = 1    # Save training samples every N epochs
        self.MAX_BATCH_POINTS = 5000       # Max number of batch samples to save
        self.BATCH_SAMPLING_SIZE = 42      # Controls how often samples and logs are generated during training
        
        # Canny Edge Detection Parameters (For pre-processing input images)
        self.CANNY_THRESHOLD_LOW = 45      # Lower threshold for Canny edge detection sensitivity
        self.CANNY_THRESHOLD_HIGH = 140    # Upper threshold for Canny edge detection strength
        
        # Data Paths (Convert to Absolute Paths)
        # Training data
        self.TRAIN_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_gt"))
        self.TRAIN_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/train_input"))
        self.TRAIN_EDGE_DIR = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/edge_maps/train"))
        self.TRAIN_GUIDANCE_DIR = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/guidance/train"))
        
        # Testing data
        self.TEST_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_gt"))
        self.TEST_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/test_input"))
        self.TEST_EDGE_DIR = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/edge_maps/test"))
        self.TEST_GUIDANCE_DIR = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/guidance/test"))
        
        # Validation data
        self.VAL_IMAGES_GT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_gt"))
        self.VAL_IMAGES_INPUT = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/val_input"))
        self.VAL_EDGE_DIR = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/edge_maps/val"))
        self.VAL_GUIDANCE_DIR = os.path.abspath(os.path.join(base_dir, "data_archive/CelebA/guidance/val"))

        #######################################################################
        # G1 MODEL CONFIGURATIONS (Edge Generator)
        #######################################################################
        
        # Output directories for G1 model artifacts
        self.MODEL_CHECKPOINT_DIR_G1 = os.path.abspath(os.path.join(base_dir, "outputs/G1/checkpoints"))
        self.EPOCH_SAMPLES_DIR_G1 = os.path.abspath(os.path.join(base_dir, "outputs/G1/generated_samples/epochs"))
        self.BATCH_SAMPLES_DIR_G1 = os.path.abspath(os.path.join(base_dir, "outputs/G1/generated_samples/batch"))
        self.LOSS_PLOT_DIR_G1 = os.path.abspath(os.path.join(base_dir, "outputs/G1/plots"))
        
        # G1 Model Path
        self.G1_MODEL_PATH = os.path.abspath(os.path.join(base_dir, "outputs/G1/checkpoints/best_edgeconnect_g1.pth"))
        
        # Optimizer Parameters for G1
        self.LEARNING_RATE_G1 = 0.0001     # Base learning rate for Adam optimizer
        self.D2G_LR_RATIO_G1 = 0.02        # Ratio between discriminator and generator learning rates
        self.BETA1 = 0.0                   # Adam optimizer beta1 parameter (momentum)
        self.BETA2 = 0.9                   # Adam optimizer beta2 parameter (RMSprop)
        self.WEIGHT_DECAY = 0.00005        # L2 regularization strength in Adam
        
        # Loss Weights for G1
        self.L1_LOSS_WEIGHT = 0.5          # Pixel-wise reconstruction loss weight
        self.ADV_LOSS_WEIGHT = 1.8         # Adversarial loss weight for generators
        self.FM_LOSS_WEIGHT = 5.5          # Feature matching loss weight


        #######################################################################
        # G2 MODEL CONFIGURATIONS (Inpainting Generator)
        #######################################################################
        
        # Output directories for G2 model artifacts
        self.MODEL_CHECKPOINT_DIR_G2 = os.path.abspath(os.path.join(base_dir, "outputs/G2/checkpoints"))
        self.EPOCH_SAMPLES_DIR_G2 = os.path.abspath(os.path.join(base_dir, "outputs/G2/generated_samples/epochs"))
        self.BATCH_SAMPLES_DIR_G2 = os.path.abspath(os.path.join(base_dir, "outputs/G2/generated_samples/batch"))
        self.LOSS_PLOT_DIR_G2 = os.path.abspath(os.path.join(base_dir, "outputs/G2/plots"))
        
        # G2 Model Path
        self.G2_MODEL_PATH = os.path.abspath(os.path.join(base_dir, "outputs/G2/checkpoints/best_edgeconnect_g2.pth"))
        
        # Optimizer Parameters for G2
        self.LEARNING_RATE_G2 = 0.0001     # Base learning rate for Adam optimizer
        self.D2G_LR_RATIO_G2 = 0.02        # Ratio between discriminator and generator learning rates
        self.BETA1_G2 = 0.0                # Adam optimizer beta1 parameter (momentum)
        self.BETA2_G2 = 0.9                # Adam optimizer beta2 parameter (RMSprop)
        self.WEIGHT_DECAY_G2 = 0.00005     # L2 regularization strength in Adam
        
        # Loss Weights for G2
        self.L1_LOSS_WEIGHT_G2 = 1.0       # Pixel-wise reconstruction loss weight
        self.ADV_LOSS_WEIGHT_G2 = 0.1      # Adversarial loss weight for generators
        self.PERCEPTUAL_LOSS_G2 = 5.5      # Perceptual loss weight (VGG19 feature loss)
        self.STYLE_LOSS_WEIGHT_G2 = 120    # Style transfer loss weight


# Initialize Config
config = Config()