# DataLoader for EdgeConnect+ G1 (Edge Generator) using Canny edge detection.
# This module handles dataset preparation for the edge generation network in the EdgeConnect+ architecture.

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import config
from utils_dl import apply_canny, dilate_mask, gen_raw_mask
from functools import lru_cache
from pathlib import Path


class EdgeConnectDataset_G1(Dataset):
    """
    Dataset class for EdgeConnect+ G1 (Edge Generator).
    
    Loads and processes images for training the edge generator network, including
    creating masked inputs, edge maps, and preparing corresponding ground truth data.
    """
    
    def __init__(self, input_dir, edge_dir, gt_dir=None, image_size=256, use_mask=True, use_gt=True, return_filenames=False):
        """
        Initialize the EdgeConnect+ G1 dataset.

        Args:
            input_dir (str): Path to masked/corrupted input images.
            edge_dir (str): Path to edge images (not used in current implementation).
            gt_dir (str, optional): Path to ground truth images. Defaults to None.
            image_size (int): Target size for images (H=W=image_size).
            use_mask (bool): Whether to include the mask in model input. Defaults to True.
            use_gt (bool): Whether to include ground truth-related processing. Defaults to True.
            return_filenames (bool): Whether to return filenames in the dataset results. Defaults to False.
        """
        self.input_dir = input_dir
        self.edge_dir = edge_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.use_mask = use_mask
        self.use_gt = use_gt
        self.return_filenames = return_filenames
        
        # Convert to Path objects for more efficient file operations
        input_path = Path(input_dir)
        self.input_files = sorted([f.name for f in input_path.glob("*.jpg")])
        
        # If using ground truth, ensure matching files exist
        if self.use_gt and gt_dir:
            gt_path = Path(gt_dir)
            self.gt_files = sorted([f.name for f in gt_path.glob("*.jpg")])
            assert len(self.input_files) == len(self.gt_files), "Mismatch in number of input and GT images."

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of images in the dataset
        """
        return len(self.input_files)
    
    @staticmethod
    @lru_cache(maxsize=32)  # Cache recently processed images to improve performance
    def _process_image(img_path):
        """
        Load and cache an image to avoid redundant disk operations.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image in BGR format with shape (H, W, 3)
        """
        return cv2.imread(img_path)

    def __getitem__(self, idx):
        """
        Get a single data sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing processed tensors with keys:
                - 'input_edge': Edge map from masked image (shape: 1 H W)
                - 'gray': Grayscale version of input image (shape: 1 H W)
                - 'mask': Binary mask where 1=valid pixels, 0=masked pixels (shape: 1 H W)
                - 'gt_edge': Ground truth edge map (shape: 1 H W, only if use_gt=True)
                - 'filenames': Original filename (str, only if return_filenames=True)
        """
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        input_img = self._process_image(input_path)  # Load masked input image

        # Generate raw binary mask from input image (1=masked/missing regions)
        raw_mask = gen_raw_mask(input_img)

        # Get dilated mask in [0,1] range where 1.0 = missing pixels
        dilated_mask_np = dilate_mask(raw_mask)

        # Generate edge maps from input image using Canny edge detection
        input_edge = apply_canny(input_img)
        
        # Remove edges at masked regions (set edge=1 where mask>0.5)
        input_edge = np.where(dilated_mask_np > 0.5, 1.0, input_edge)

        # Convert input image to grayscale and normalize to [0,1] range
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.astype(np.float32) / 255.0

        # Convert numpy arrays to PyTorch tensors with correct dimensions (C×H×W)
        gray_img_tensor = torch.from_numpy(gray_img).float().unsqueeze(0)  # Shape: (1, H, W)
        input_edge_tensor = torch.from_numpy(input_edge).float().unsqueeze(0)  # Shape: (1, H, W)
        
        # Invert the mask for the model (1=valid pixels, 0=masked pixels)
        mask_for_model = 1.0 - dilated_mask_np
        mask_tensor = torch.from_numpy(mask_for_model).float().unsqueeze(0)  # Shape: (1, H, W)

        # Prepare result dictionary with common elements
        result = {
            "input_edge": input_edge_tensor,  # Shape: (1, H, W)
            "gray": gray_img_tensor,          # Shape: (1, H, W)
            "mask": mask_tensor               # Shape: (1, H, W)
        }

        # If GT is enabled, add ground truth edge map
        if self.use_gt and self.gt_dir:
            gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
            gt_img = self._process_image(gt_path)  # Load ground truth image
            gt_edge = apply_canny(gt_img)  # Extract edges from ground truth image
            result["gt_edge"] = torch.from_numpy(gt_edge).float().unsqueeze(0)  # Shape: (1, H, W)

        # Include filename in results if requested
        if self.return_filenames:
            result["filenames"] = self.input_files[idx]

        return result


def get_dataloader_g1(split="train", batch_size=config.BATCH_SIZE_G1, shuffle=True, use_mask=True, use_gt=True, return_filenames=False):
    """
    Create a DataLoader for the EdgeConnect+ G1 (Edge Generator) model.
    
    Args:
        split (str): Dataset split to use ('train', 'test', 'val', or 'demo').
        batch_size (int): Number of samples in each batch. Defaults to config.BATCH_SIZE_G1.
        shuffle (bool): Whether to shuffle the data. Defaults to True.
        use_mask (bool): Whether to include the mask in the model input. Defaults to True.
        use_gt (bool): Whether to include ground truth data. Defaults to True.
        return_filenames (bool): Whether to return filenames in results. Defaults to False.
    
    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset split
                   with batch size, shuffle, and worker settings from config.
    
    Raises:
        ValueError: If an invalid dataset split is specified.
    """
    # Map dataset splits to their corresponding input and ground truth paths
    dataset_paths = {
        "train": (config.TRAIN_IMAGES_INPUT, config.TRAIN_IMAGES_GT),
        "test": (config.TEST_IMAGES_INPUT, config.TEST_IMAGES_GT),
        "val": (config.VAL_IMAGES_INPUT, config.VAL_IMAGES_GT),
        "demo": (config.DEMO_IMAGES_INPUT, config.DEMO_IMAGES_GT)
    }
    
    # Validate the requested dataset split
    if split not in dataset_paths:
        raise ValueError("Invalid dataset split. Choose from 'train', 'test', 'val', or 'demo'.")
    
    # Get the appropriate paths for the selected split
    input_path, gt_path = dataset_paths[split]
    
    # Initialize the dataset with the specified configuration
    dataset = EdgeConnectDataset_G1(
        input_dir=input_path,
        edge_dir=None,  # Edge directory is not used in current implementation
        gt_dir=gt_path if use_gt else None,
        image_size=config.IMAGE_SIZE,
        use_mask=use_mask,
        use_gt=use_gt,
        return_filenames=return_filenames
    )
    
    # Create and return the DataLoader with proper configuration
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY, 
        prefetch_factor=2  # Pre-fetch 2 batches per worker for efficiency
    )