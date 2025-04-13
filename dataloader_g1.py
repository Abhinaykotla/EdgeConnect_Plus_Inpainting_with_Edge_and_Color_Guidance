# dataloader.py: DataLoader for EdgeConnect+ G1 (Edge Generator) using Canny edge detection.

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

def dilate_mask(mask, kernel_size=5, iterations=1):
    """
    Dilate the binary mask.
    
    Args:
        mask: Input mask tensor or numpy array. Should be in format where:
              0 = missing pixels (value < 10)
              255 = known pixels (value >= 10)
        kernel_size: Size of dilation kernel
        iterations: Number of dilation iterations
    
    Returns:
        Dilated mask as a numpy array where:
        1.0 = missing pixels
        0.0 = known pixels
    """
    # Convert mask to numpy if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze().cpu().numpy()  # Shape: (H, W)
    else:
        mask_np = mask.squeeze() if hasattr(mask, 'squeeze') else mask
    
    # Make sure mask is in the proper format [0-255] with 0 for missing pixels
    # If mask is in [0,1] range, convert to [0,255]
    if mask_np.max() <= 1.0 and np.min(mask_np) >= 0:
        mask_np = mask_np * 255

    # Binary mask: 1 for missing pixels (where mask == 0), 0 for known pixels
    binary_mask = (mask_np < 10).astype(np.float32)
    
    # Dilate the binary mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    dilated_mask_uint8 = cv2.dilate(binary_mask_uint8, kernel, iterations=iterations)
    
    # Convert to [0.0, 1.0] range where 1.0 = missing pixels
    dilated_mask = dilated_mask_uint8.astype(np.float32) / 255.0
    
    return dilated_mask

def remove_mask_edge(mask, img):
    """
    Remove the edges of the mask in the edge image by painting white (1.0)
    where the dilated mask indicates missing regions.
    
    Args:
        mask: Input mask tensor or numpy array
        img: Edge image tensor or numpy array
    
    Returns:
        Edge image with mask edges removed
    """
    # Convert img to numpy if it's a tensor
    if isinstance(img, torch.Tensor):
        img_np = img.squeeze().cpu().numpy()  # Shape: (H, W)
    else:
        img_np = img.squeeze() if hasattr(img, 'squeeze') else img
    
    # Get dilated mask
    dilated_mask = dilate_mask(mask)
    
    # Paint white (1.0) where dilated mask indicates missing pixels
    result = np.where(dilated_mask > 0.5, 1.0, img_np)
    
    return result

def extract_mask(input_img):
    """
    Extract the binary mask from the input image.
    
    Args:
        input_img (numpy.ndarray): Input image (H, W, C) in RGB format.
    
    Returns:
        numpy.ndarray: Inverted binary mask (H, W) where 0 = missing pixels, 255 = known pixels.
    """
    # Consider pixels as missing if all RGB values > 245
    mask_binary = np.all(input_img > 245, axis=-1).astype(np.float32)  # Shape: (H, W)
    raw_mask = 255 - mask_binary * 255  # Invert mask (0s for missing pixels, 255s for known pixels)
    return raw_mask

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

        # Extract mask using the new function
        raw_mask = extract_mask(input_img)

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