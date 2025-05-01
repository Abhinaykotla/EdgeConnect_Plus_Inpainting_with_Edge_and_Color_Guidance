# dataloader.py: DataLoader for EdgeConnect+ G1 (Edge Generator) using Canny edge detection.

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
    def __init__(self, input_dir, edge_dir, gt_dir=None, image_size=256, use_mask=True, use_gt=True, return_filenames=False):
        """
        Dataset loader for EdgeConnect+ G1 (Edge Generator).

        Args:
            input_dir (str): Path to masked images.
            edge_dir (str): Path to edge images.
            gt_dir (str, optional): Path to ground truth images. Defaults to None.
            image_size (int): Size of images.
            use_mask (bool): Whether to include the mask as input.
            use_gt (bool): Whether to include ground truth-related processing. Defaults to True.
            return_filenames (bool): Whether to return filenames in the dataset. Defaults to False.
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
        
        if self.use_gt and gt_dir:
            gt_path = Path(gt_dir)
            self.gt_files = sorted([f.name for f in gt_path.glob("*.jpg")])
            assert len(self.input_files) == len(self.gt_files), "Mismatch in number of input and GT images."

    def __len__(self):
        return len(self.input_files)
    
    @staticmethod
    @lru_cache(maxsize=32)  # Cache recently processed images
    def _process_image(img_path):
        """Process and cache image loading to avoid redundant operations"""
        return cv2.imread(img_path)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        input_img = self._process_image(input_path)  # Masked Image

        # Generate raw mask from input image
        raw_mask = gen_raw_mask(input_img)

        # Get dilated mask (in [0,1] range where 1.0 = missing pixels)
        dilated_mask_np = dilate_mask(raw_mask)

        # Generate Edge Maps
        input_edge = apply_canny(input_img)  # Edges from masked image
        input_edge = np.where(dilated_mask_np > 0.5, 1.0, input_edge)  # Remove edge at masked regions

        # Convert Grayscale Image from input_img
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert to Tensors - use a single operation to convert to Float32 and add dimension
        gray_img_tensor = torch.from_numpy(gray_img).float().unsqueeze(0)  # Shape: (1, H, W)
        input_edge_tensor = torch.from_numpy(input_edge).float().unsqueeze(0)  # Shape: (1, H, W)
        mask_for_model = 1.0 - dilated_mask_np
        mask_tensor = torch.from_numpy(mask_for_model).float().unsqueeze(0)  # Shape: (1, H, W)

        # Common elements for both return paths
        result = {
            "input_edge": input_edge_tensor,  # Shape: (1, H, W)
            "gray": gray_img_tensor,          # Shape: (1, H, W)
            "mask": mask_tensor               # Shape: (1, H, W)
        }

        # If GT is enabled, add GT-related data
        if self.use_gt and self.gt_dir:
            gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
            gt_img = self._process_image(gt_path)  # Ground Truth Image
            gt_edge = apply_canny(gt_img)  # Edges from ground truth image
            result["gt_edge"] = torch.from_numpy(gt_edge).float().unsqueeze(0)  # Shape: (1, H, W)

        # If return_filenames is enabled, add filenames to the result
        if self.return_filenames:
            result["filenames"] = self.input_files[idx]

        return result


def get_dataloader_g1(split="train", batch_size=config.BATCH_SIZE_G1, shuffle=True, use_mask=True, use_gt=True, return_filenames=False):
    """
    Initializes the DataLoader for EdgeConnect+ G1 (Edge Generator).

    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').
        batch_size (int, optional): Batch size for the dataloader. Defaults to config value.
        shuffle (bool): Whether to shuffle the data. Defaults to True.
        use_mask (bool): Whether to include the mask as input.
        use_gt (bool): Whether to include ground truth-related processing.
        return_filenames (bool): Whether to return filenames in the dataset. Defaults to False.

    Returns:
        DataLoader: A PyTorch DataLoader for the specified dataset split.
    """
    dataset_paths = {
        "train": (config.TRAIN_IMAGES_INPUT, config.TRAIN_IMAGES_GT),
        "test": (config.TEST_IMAGES_INPUT, config.TEST_IMAGES_GT),
        "val": (config.VAL_IMAGES_INPUT, config.VAL_IMAGES_GT)
    }
    
    if split not in dataset_paths:
        raise ValueError("Invalid dataset split. Choose from 'train', 'test', or 'val'.")
    
    input_path, gt_path = dataset_paths[split]
    dataset = EdgeConnectDataset_G1(
        input_dir=input_path,
        edge_dir=None,  # Edge directory is not used in this implementation
        gt_dir=gt_path if use_gt else None,
        image_size=config.IMAGE_SIZE,
        use_mask=use_mask,
        use_gt=use_gt,
        return_filenames=return_filenames  # Pass the new parameter
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY, 
        prefetch_factor=2
    )