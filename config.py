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
        self.BATCH_SIZE = 10 
        self.NUM_WORKERS = 4  
        self.EPOCHS = 100  
        self.EARLY_STOP_PATIENCE = 5  # Updated for faster tracking
        self.TRAINING_SAMPLE_EPOCHS = 2  # Updated for faster tracking
        self.VALIDATION_SAMPLE_EPOCHS = 5  # Updated for faster tracking
        self.IMAGE_SIZE = 256  
        self.LEARNING_RATE = 0.0001  
        self.D2G_LR_RATIO = 0.05  # Reduced to slow down D1
        self.BETA1 = 0.5  
        self.BETA2 = 0.999  
        self.WEIGHT_DECAY = 0.00005  # Reduced for stability
        self.MAX_ITERS = 2000000  

        # System Settings
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = 42  
        self.GPU_IDS = [0]  
        self.DEBUG = 0  
        self.VERBOSE = 1
        self.MODEL_CHECKPOINT_DIR = os.path.abspath(os.path.join(base_dir, "models/checkpoints"))

        # Loss Weights (Optimized)
        self.L1_LOSS_WEIGHT = 2.0  # Reduced for better stability
        self.ADV_LOSS_WEIGHT = 0.2  # Increased slightly to encourage learning
        self.FM_LOSS_WEIGHT = 10.0  
        self.STYLE_LOSS_WEIGHT = 1.0  
        self.CONTENT_LOSS_WEIGHT = 1.0  

        # GAN Settings
        self.GAN_LOSS = "nsgan"  
        self.GAN_POOL_SIZE = 0  

        # Edge Detection Parameters
        self.EDGE_THRESHOLD = 0.5  
        self.SIGMA = 2  

        # Logging & Checkpoints
        self.SAVE_IMAGES_EVERY = 5  
        self.SAVE_MODEL_EVERY = 10  
        self.OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, "output"))  

# Initialize Config
config = Config()
