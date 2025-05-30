# g2_dataloader.py - Dataset loader for the second generator (G2) in EdgeConnect+ architecture
# Handles loading, processing, and batching of images with their guidance and mask information

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config
from utils_dl import dilate_mask, gen_raw_mask, validate_guidance_images
from functools import lru_cache
from pathlib import Path


class EdgeConnectDataset_G2(Dataset):
    def __init__(self, input_dir, guidance_dir, gt_dir=None, image_size=256, use_gt=True):
        """
        Dataset loader for EdgeConnect+ G2.

        Args:
            input_dir (str): Path to input images.
            guidance_dir (str): Path to guidance images.
            gt_dir (str, optional): Path to ground truth images. Defaults to None.
            image_size (int): Size of images.
            use_gt (bool): Whether to include ground truth-related processing. Defaults to True.
        """
        self.input_dir = input_dir
        self.guidance_dir = guidance_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.use_gt = use_gt
        
        # Convert to Path objects for more efficient file operations
        input_path = Path(input_dir)
        guidance_path = Path(guidance_dir)
        self.input_files = sorted([f.name for f in input_path.glob("*.jpg")])
        self.guidance_files = sorted([f.name for f in guidance_path.glob("*.jpg")])

        # Verify dataset integrity by checking file counts match
        if self.use_gt and gt_dir:
            gt_path = Path(gt_dir)
            self.gt_files = sorted([f.name for f in gt_path.glob("*.jpg")])
            assert len(self.input_files) == len(self.guidance_files) == len(self.gt_files), \
                "Mismatch in number of images."
        else:
            assert len(self.input_files) == len(self.guidance_files), \
                "Mismatch in number of input and guidance images."
        
        # Define transform for consistent resizing
        # Converts images to PyTorch tensors and resizes them to the target dimensions
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size))
        ])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.input_files)
    
    @staticmethod
    @lru_cache(maxsize=128)  # Cache recently processed images to improve performance
    def _process_image(img_path):
        """
        Process and cache image loading to avoid redundant operations.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: RGB image as a numpy array, or None if loading failed
        """
        img = cv2.imread(img_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB color space
        return None

    def __getitem__(self, idx):
        """
        Fetches and processes a single data sample.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing:
                - input_img (tensor): Input image tensor of shape [3, H, W]
                - guidance_img (tensor): Guidance image tensor of shape [3, H, W]
                - mask (tensor): Binary mask tensor of shape [1, H, W]
                - gt_img (tensor, optional): Ground truth image tensor of shape [3, H, W]
        """
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        guidance_path = os.path.join(self.guidance_dir, self.guidance_files[idx])

        # Read RGB images using cached method
        input_img = self._process_image(input_path)
        guidance_img = self._process_image(guidance_path)

        # Generate raw mask from input image
        # Raw mask identifies missing regions in the input image
        raw_mask = gen_raw_mask(input_img)

        # Get dilated mask (in [0,1] range where 1.0 = missing pixels)
        # Dilation expands the mask slightly to ensure all damaged regions are covered
        dilated_mask_np = dilate_mask(raw_mask)

        # Convert to Tensors with proper formatting
        input_tensor = self.transform(input_img)
        guidance_tensor = self.transform(guidance_img)
        # Invert mask for model use (0 = missing pixels, 1 = valid pixels)
        mask_for_model = 1.0 - dilated_mask_np  
        mask_tensor = torch.from_numpy(mask_for_model).float().unsqueeze(0)  # Shape: (1, H, W)
        
        # Common return elements
        result = {
            "input_img": input_tensor,
            "guidance_img": guidance_tensor,
            "mask": mask_tensor
        }

        # If ground truth is enabled, process GT-related data
        if self.use_gt and self.gt_dir:
            gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
            gt_img = self._process_image(gt_path)
            result["gt_img"] = self.transform(gt_img)

        return result


def get_dataloader_g2(split="train", batch_size=config.BATCH_SIZE_G2, shuffle=True, use_gt=True):
    """
    Returns a DataLoader for the G2 dataset based on the specified split.
    Validates and generates guidance images if needed.

    Args:
        split (str): Dataset split to use ('train', 'test', 'val', or 'demo').
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        use_gt (bool): Whether to include ground truth-related processing. Defaults to True.

    Returns:
        DataLoader: PyTorch DataLoader for the G2 dataset.
    """
    # First, validate guidance images - this will generate them if needed
    validate_guidance_images(split)

    # Use dictionary mapping for more efficient directory selection
    # Maps dataset splits to their corresponding directories
    dataset_paths = {
        "train": (config.TRAIN_IMAGES_INPUT, config.TRAIN_GUIDANCE_DIR, config.TRAIN_IMAGES_GT),
        "test": (config.TEST_IMAGES_INPUT, config.TEST_GUIDANCE_DIR, config.TEST_IMAGES_GT),
        "val": (config.VAL_IMAGES_INPUT, config.VAL_GUIDANCE_DIR, config.VAL_IMAGES_GT),
        "demo": (config.DEMO_IMAGES_INPUT, config.DEMO_GUIDANCE_DIR, config.DEMO_IMAGES_GT)
    }
    
    if split not in dataset_paths:
        raise ValueError("Invalid split. Choose from 'train', 'test', 'val', or 'demo'.")
    
    input_dir, guidance_dir, gt_dir = dataset_paths[split]
    
    # Create the dataset with appropriate directories based on the requested split
    dataset = EdgeConnectDataset_G2(
        input_dir=input_dir,
        guidance_dir=guidance_dir,
        gt_dir=gt_dir if use_gt else None,
        image_size=config.IMAGE_SIZE,
        use_gt=use_gt
    )

    # Return the DataLoader with the configured options
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE_G2,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,  # Parallel data loading threads
        pin_memory=config.PIN_MEMORY,    # Accelerates transfer to GPU if used
        prefetch_factor=2                # Number of batches to prefetch
    )


if __name__ == '__main__':
    # Example usage of the dataloader for testing/debugging purposes
    train_loader = get_dataloader_g2(split="train")

    for batch in train_loader:
        input_img = batch["input_img"]
        guidance_img = batch["guidance_img"]
        mask = batch["mask"]
        gt_img = batch.get("gt_img", None)  # Optional ground truth image

        # Process your batch here
        print(f"INFO: Input shape: {input_img.shape}, Guidance shape: {guidance_img.shape}, Mask shape: {mask.shape}")
        if gt_img is not None:
            print(f"INFO: GT shape: {gt_img.shape}")
