# dataloader.py: DataLoader for EdgeConnect+ G1 (Edge Generator) using Canny edge detection.

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import config
from utils_dl import apply_canny, dilate_mask, gen_raw_mask


class EdgeConnectDataset_G1(Dataset):
    def __init__(self, input_dir, gt_dir, image_size=256, use_mask=False):
        """
        Dataset loader for EdgeConnect+ G1 (Edge Generator).
        
        Args:
            input_dir (str): Path to masked images.
            gt_dir (str): Path to ground truth images.
            image_size (int): Size of images.
            use_mask (bool): Whether to include the mask as input.
        """
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.use_mask = use_mask

        self.input_files = sorted([f.name for f in os.scandir(input_dir) if f.name.endswith('.jpg')])
        self.gt_files = sorted([f.name for f in os.scandir(gt_dir) if f.name.endswith('.jpg')])
        
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        # Load Images
        input_img = cv2.imread(input_path)  # Masked Image
        gt_img = cv2.imread(gt_path)        # Ground Truth Image

        raw_mask = gen_raw_mask(input_img)  # Generate raw mask from input image

        # Get dilated mask (in [0,1] range where 1.0 = missing pixels)
        dilated_mask_np = dilate_mask(raw_mask)

        # Generate Edge Maps
        input_edge = apply_canny(input_img)  # Edges from masked image
        input_edge = np.where(dilated_mask_np > 0.5, 1.0, input_edge)  # Remove edge at masked regions
        gt_edge = apply_canny(gt_img)  # Edges from ground truth image

        # Convert Grayscale Image from input_img
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        gray_img_tensor = torch.from_numpy(gray_img).float().unsqueeze(0)  # Shape: (1, H, W)

        # Convert to Tensors
        input_edge = torch.from_numpy(input_edge).float().unsqueeze(0)  # Shape: (1, H, W)
        gt_edge = torch.from_numpy(gt_edge).float().unsqueeze(0)        # Shape: (1, H, W)
        mask_for_model = 1.0 - dilated_mask_np
        mask_for_model = torch.from_numpy(mask_for_model).float().unsqueeze(0)  # Shape: (1, H, W)

        # Return all components
        return {
            "input_edge": input_edge,     # Shape: (1, H, W)
            "gt_edge": gt_edge,           # Shape: (1, H, W)
            "gray": gray_img_tensor,      # Shape: (1, H, W)
            "mask": mask_for_model        # Shape: (1, H, W)
        }



# Initialize DataLoader for G1 with optional mask input
def get_dataloader_g1(split="train", use_mask=False):
    dataset_paths = {
        "train": (config.TRAIN_IMAGES_INPUT, config.TRAIN_IMAGES_GT),
        "test": (config.TEST_IMAGES_INPUT, config.TEST_IMAGES_GT),
        "val": (config.VAL_IMAGES_INPUT, config.VAL_IMAGES_GT)
    }
    if split not in dataset_paths:
        raise ValueError("Invalid dataset split. Choose from 'train', 'test', or 'val'.")
    
    input_path, gt_path = dataset_paths[split]
    dataset = EdgeConnectDataset_G1(input_path, gt_path, config.IMAGE_SIZE, use_mask)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, prefetch_factor=2)

# if __name__ == "__main__":
#     # Test DataLoader
#     dataloader = get_dataloader_g1(split="val", use_mask=True)
#     for batch in dataloader:
#         print(batch["input_edge"].shape, batch["gt_edge"].shape, batch["mask"].shape, batch["gray"].shape)
#         break