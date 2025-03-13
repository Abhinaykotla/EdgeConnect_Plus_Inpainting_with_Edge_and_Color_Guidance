import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob

# Function to apply Canny edge detection
def apply_canny(image):
    """
    Apply Canny edge detection to an image and invert edges 
    to get a white background with black edges.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    image = np.transpose(image, (1, 2, 0)) if image.shape[0] == 3 else image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    edges = cv2.Canny(image, 50, 150)
    edges = (255 - edges).astype(np.float32) / 255.0  # Normalize

    return torch.from_numpy(edges).unsqueeze(0)  # Add channel dimension



# Function to fetch a random mask
def get_random_mask(mask_dir):
    """Fetches a random mask from the mask directory and ensures it's properly binary (0s and 1s)."""
    mask_paths = glob(os.path.join(mask_dir, "*.png"))
    mask_path = random.choice(mask_paths)  # Pick a random mask

    # Load the mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert 255 values to 1 (Normalize to binary format)
    mask = (mask > 127).astype(np.float32)  # Ensures mask is only 0 or 1
    mask = 1 - mask  # Invert mask (0s are missing regions, 1s are visible regions)
    return mask

def apply_mask_to_image(image, mask):
    """
    Applies a mask to an RGB image. 
    - Removes areas where the mask is active (1) and replaces them with white (255,255,255).
    - Keeps areas where the mask is inactive (0).

    Args:
        image (numpy.ndarray): RGB image of shape (256, 256, 3)
        mask (numpy.ndarray): Binary mask of shape (256, 256), values in {0,1}

    Returns:
        numpy.ndarray: Masked image with white missing areas.
    """

    # Ensure the mask is binary (0 and 1)
    mask = (mask > 0).astype(np.float32)  # Convert any nonzero values to 1

    # Expand mask dimensions to match the image shape (256,256) → (256,256,3)
    mask = np.expand_dims(mask, axis=-1)

    # Set masked areas to white (255,255,255)
    masked_image = image * (1 - mask) + mask * 255  # Keeps areas where mask is 0, replaces masked areas with white

    return masked_image.astype(np.uint8)  # Convert back to uint8 format


def mask_edges(edges_original, mask):
    """
    Apply a mask to an existing edge map to remove edges from missing areas.

    Args:
        edges_original (numpy.ndarray or torch.Tensor): Edge map of the original image, shape (H, W)
        mask (numpy.ndarray or torch.Tensor): Binary mask of missing areas (1=missing, 0=valid)

    Returns:
        torch.Tensor: Masked edge map with missing area edges removed.
    """

    # Convert to numpy if input is a tensor
    if isinstance(edges_original, torch.Tensor):
        edges_np = edges_original.cpu().numpy()
    else:
        edges_np = np.array(edges_original)

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)

    # Ensure mask is binary (0 = valid, 1 = missing)
    mask_np = (mask_np > 0).astype(np.uint8)

    # Apply mask to remove edges in missing areas
    edges_masked = edges_np * (1 - mask_np)  # Keeps edges where mask is 0, removes where mask is 1

    # Convert back to torch tensor
    edges_masked = torch.tensor(edges_masked, dtype=torch.float32)

    return edges_masked



# Function to apply Gaussian blur to generate color maps
def apply_gaussian_blur(image, kernel_size=15):
    """Applies Gaussian blur to propagate color information into missing regions."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Custom Dataset for G1 Training (Edge Generator)
class EdgeGeneratorDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = glob(os.path.join(image_dir, "*.jpg"))
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads an image, applies a random mask, and generates edge maps for G1 training."""

        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Fetch a random mask
        mask = get_random_mask(self.mask_dir)

        # Apply mask to the image (hide missing regions)
        masked_image = apply_mask_to_image(image, mask)

        # Generate edges using Canny
        edges_original = apply_canny(image)
        # edges_masked = mask_edges(edges_original, mask)
        # masked_image = image * (1 - mask) + mask * 255
        edges_masked = edges_original * (1 - mask) + mask * 1

        # Convert to PyTorch tensors
        masked_image = self.transform(masked_image)
        original_image = self.transform(image)

        return masked_image, edges_masked, edges_original, original_image

# Custom Dataset for G2 Training (Final Inpainting)
class InpaintingGeneratorDataset(Dataset):
    pass

# Function to create DataLoader for G1 (Edge Generator Training)
def get_dataloader_g1(image_dir, mask_dir, batch_size=8, num_workers=4, shuffle=True):
    dataset = EdgeGeneratorDataset(image_dir, mask_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=min(2, batch_size // 2),
        persistent_workers=num_workers > 0
    )

# Function to create DataLoader for G2 (Final Inpainting Model Training)

