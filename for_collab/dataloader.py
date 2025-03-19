# dataloader.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import config

def apply_canny(image):
    """
    Apply Canny edge detection to an image.
    """
    # Ensure image is in the right format for Canny
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
    # Apply Canny edge detection
    edges = cv2.Canny(image, config.CANNY_THRESHOLD_LOW, config.CANNY_THRESHOLD_HIGH) # Shape: (H, W)
    
    # Invert and normalize to [0, 1] for the edge map
    edges = (255 - edges).astype(np.float32) / 255.0
    return edges

def remove_mask_egde(mask, img):
    """
    Remove the edges of the mask in the edge image by painting white (1.0)
    where the dilated mask indicates missing regions.
    """
    # Convert mask to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze().cpu().numpy()  # Shape: (H, W)
    else:
        mask_np = mask.squeeze() if hasattr(mask, 'squeeze') else mask
    
    # Convert img to numpy if it's a tensor
    if isinstance(img, torch.Tensor):
        img_np = img.squeeze().cpu().numpy()  # Shape: (H, W)
    else:
        img_np = img.squeeze() if hasattr(img, 'squeeze') else img
    
    # Make sure mask is in the proper format [0-255] with 0 for missing pixels
    # If mask is in [0,1] range with 0 for missing pixels, convert to [0,255]
    if mask_np.max() <= 1.0 and np.min(mask_np) >= 0:
        mask_np = mask_np * 255

    # Binary mask: 1 for missing pixels (where mask == 0), 0 for known pixels
    binary_mask = (mask_np < 10).astype(np.float32)
    
    # Dilate the binary mask
    kernel = np.ones((5, 5), np.uint8)
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    dilated_mask_uint8 = cv2.dilate(binary_mask_uint8, kernel, iterations=1)
    dilated_mask = dilated_mask_uint8.astype(np.float32) / 255.0
    
    # Paint white (1.0) where dilated mask indicates missing pixels
    result = np.where(dilated_mask > 0.5, 1.0, img_np)
    
    return result

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
        gt_img = cv2.imread(gt_path)  # Ground Truth Image

        # # Resize Images
        # input_img = cv2.resize(input_img, (self.image_size, self.image_size))
        # gt_img = cv2.resize(gt_img, (self.image_size, self.image_size))

        # Extract mask: Consider pixels as missing if all RGB values > 245
        mask_binary = np.all(input_img > 245, axis=-1).astype(np.float32) # Shape: (H, W)
        mask = 255 - mask_binary * 255  # Invert mask (0s for missing pixels, 255s for known pixels)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Shape: (1, H, W)

        # Generate Edge Maps
        input_edge = apply_canny(input_img)  # Edges from masked image
        
        # Apply mask edge removal
        input_edge = remove_mask_egde(mask, input_edge)

        gt_edge = apply_canny(gt_img)  # Edges from ground truth image

        # Convert to Tensors if not already
        if not isinstance(input_edge, torch.Tensor):
            input_edge = torch.from_numpy(input_edge).float().unsqueeze(0)  # Shape: (1, H, W)
        
        if not isinstance(gt_edge, torch.Tensor):
            gt_edge = torch.from_numpy(gt_edge).float().unsqueeze(0)  # Shape: (1, H, W)\

        # Return full-res and resized ground truth edges
        if self.use_mask:
            mask = mask / 255.0  # Normalize mask to [0, 1]
            return {
                "input_edge": input_edge, 
                "gt_edge": gt_edge,  # Full-size GT edges
                # "gt_edge_resized": gt_edge_resized,  # Resized for Discriminator
                "mask": mask
            }
        else:
            return {
                "input_edge": input_edge,
                "gt_edge": gt_edge,  
                # "gt_edge_resized": gt_edge_resized
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
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, prefetch_factor=2)

# if __name__ == "__main__":
#     # Test DataLoader
#     dataloader = get_dataloader_g1(split="val", use_mask=True)
#     for batch in dataloader:
#         print(batch["input_edge"].shape, batch["gt_edge"].shape, batch["mask"].shape)
#         break