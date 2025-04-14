import cv2
import torch
import numpy as np
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