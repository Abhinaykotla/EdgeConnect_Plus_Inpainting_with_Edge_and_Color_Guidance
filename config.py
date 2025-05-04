import torch
import os

class Config:
    def __init__(self):
        # Detect Colab
        try:
            in_colab = "google.colab" in str(get_ipython()) # type: ignore[no-untyped-call]
        except NameError:
            in_colab = False

        if "COLAB_GPU" in os.environ or in_colab:
            # Paths for Colab
            self.BASE_DIR = "/content/EdgeConnectPlus"
            self.DRIVE_DS_DIR = "/content/drive/MyDrive/EdgeConnectPlus/"
            self.OUTPUT_DIR = "/content/drive/MyDrive/EdgeConnectPlus/outputs"
        else:
            # Local environment
            self.BASE_DIR = os.getcwd()
            self.DRIVE_DS_DIR = os.getcwd()
            self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "outputs")

        #######################################################################
        # GENERAL CONFIGURATIONS (Shared between G1 and G2)
        #######################################################################
        
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 12
        self.PIN_MEMORY = True
        self.EPOCHS = 45
        self.EARLY_STOP_PATIENCE = 5
        self.IMAGE_SIZE = 256

        self.VALIDATION_SAMPLE_EPOCHS = 5
        self.TRAINING_SAMPLE_EPOCHS = 1
        self.MAX_BATCH_POINTS = 300000
        self.BATCH_SAMPLING_SIZE = 15 # 200

        self.CANNY_THRESHOLD_LOW = 110
        self.CANNY_THRESHOLD_HIGH = 180
        self.OVERRIDE_LR = True
        self.MAX_CHECKPOINTS = 10

        #######################################################################
        # DATASET PATHS (you can limit to only train data in Colab)
        #######################################################################
        self.TRAIN_IMAGES_GT = os.path.join(self.BASE_DIR, "data_archive/CelebA/train_gt")
        self.TRAIN_IMAGES_INPUT = os.path.join(self.BASE_DIR, "data_archive/CelebA/train_input")
        self.TRAIN_EDGE_DIR = os.path.join(self.BASE_DIR, "data_archive/CelebA/edge_maps/train")
        self.TRAIN_GUIDANCE_DIR = os.path.join(self.BASE_DIR, "data_archive/CelebA/guidance/train")

        self.TEST_IMAGES_GT = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/test_gt")
        self.TEST_IMAGES_INPUT = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/test_input")
        self.TEST_EDGE_DIR = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/edge_maps/test")
        self.TEST_GUIDANCE_DIR = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/guidance/test")

        self.VAL_IMAGES_GT = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/val_gt")
        self.VAL_IMAGES_INPUT = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/val_input")
        self.VAL_EDGE_DIR = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/edge_maps/val")
        self.VAL_GUIDANCE_DIR = os.path.join(self.DRIVE_DS_DIR, "data_archive/CelebA/guidance/val")

        #######################################################################
        # G1 MODEL CONFIGURATIONS
        #######################################################################
        self.MODEL_CHECKPOINT_DIR_G1 = os.path.join(self.OUTPUT_DIR, "G1/checkpoints")
        self.EPOCH_SAMPLES_DIR_G1 = os.path.join(self.OUTPUT_DIR, "G1/generated_samples/epochs")
        self.BATCH_SAMPLES_DIR_G1 = os.path.join(self.OUTPUT_DIR, "G1/generated_samples/batch")
        self.LOSS_PLOT_DIR_G1 = os.path.join(self.OUTPUT_DIR, "G1/plots")
        self.G1_MODEL_PATH = os.path.join(self.BASE_DIR, "best_models/g1_best.pth")

        self.BATCH_SIZE_G1 = 20 # 70
        self.BATCH_SIZE_G1_INFERENCE = 94
        self.LEARNING_RATE_G1 = 0.0001
        self.D2G_LR_RATIO_G1 = 0.3
        self.BETA1 = 0.0
        self.BETA2 = 0.9
        self.WEIGHT_DECAY = 0.00005

        self.L1_LOSS_WEIGHT = 1.0
        self.ADV_LOSS_WEIGHT = 2.5
        self.FM_LOSS_WEIGHT = 2.0
        self.VGG_LOSS_WEIGHT = 0.5


        #######################################################################
        # G2 MODEL CONFIGURATIONS
        #######################################################################
        self.MODEL_CHECKPOINT_DIR_G2 = os.path.join(self.OUTPUT_DIR, "G2/checkpoints")
        self.EPOCH_SAMPLES_DIR_G2 = os.path.join(self.OUTPUT_DIR, "G2/generated_samples/epochs")
        self.BATCH_SAMPLES_DIR_G2 = os.path.join(self.OUTPUT_DIR, "G2/generated_samples/batch")
        self.LOSS_PLOT_DIR_G2 = os.path.join(self.OUTPUT_DIR, "G2/plots")
        self.G2_MODEL_PATH = os.path.join(self.BASE_DIR, "best_models/g2_best.pth")

        self.BATCH_SIZE_G2 = 18
        self.BATCH_SIZE_G2_INFERENCE = 64
        self.LEARNING_RATE_G2 = 0.0001
        self.D2G_LR_RATIO_G2 = 0.5
        self.BETA1_G2 = 0.0
        self.BETA2_G2 = 0.9
        self.WEIGHT_DECAY_G2 = 0.00005
        self.GRADIENT_PENALTY_WEIGHT_G2 = 0.3

        self.L1_LOSS_WEIGHT_G2 = 1.0
        self.ADV_LOSS_WEIGHT_G2 = 0.1
        self.PERCEPTUAL_LOSS_G2 = 3.0
        self.STYLE_LOSS_WEIGHT_G2 = 2
        self.FM_LOSS_WEIGHT_G2 = 0.5


# Global config instance
config = Config()
