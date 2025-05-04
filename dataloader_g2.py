# g2_dataloader.py

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

        if self.use_gt and gt_dir:
            gt_path = Path(gt_dir)
            self.gt_files = sorted([f.name for f in gt_path.glob("*.jpg")])
            assert len(self.input_files) == len(self.guidance_files) == len(self.gt_files), \
                "Mismatch in number of images."
        else:
            assert len(self.input_files) == len(self.guidance_files), \
                "Mismatch in number of input and guidance images."
        
        # Define transform for consistent resizing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size))
        ])

    def __len__(self):
        return len(self.input_files)
    
    @staticmethod
    @lru_cache(maxsize=32)  # Cache recently processed images
    def _process_image(img_path):
        """Process and cache image loading to avoid redundant operations"""
        img = cv2.imread(img_path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        guidance_path = os.path.join(self.guidance_dir, self.guidance_files[idx])

        # Read RGB images using cached method
        input_img = self._process_image(input_path)
        guidance_img = self._process_image(guidance_path)

        # Generate raw mask from input image
        raw_mask = gen_raw_mask(input_img)

        # Get dilated mask (in [0,1] range where 1.0 = missing pixels)
        dilated_mask_np = dilate_mask(raw_mask)

        # Convert to Tensors with proper formatting
        input_tensor = self.transform(input_img)
        guidance_tensor = self.transform(guidance_img)
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
        split (str): Dataset split to use ('train', 'test', or 'val').
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        use_gt (bool): Whether to include ground truth-related processing. Defaults to True.

    Returns:
        DataLoader: PyTorch DataLoader for the G2 dataset.
    """
    # First, validate guidance images - this will generate them if needed
    successful = validate_guidance_images(split)
    
    # If validation wasn't successful, it means images are still being generated
    # or there was an error. Make sure they're generated before continuing.
    if not successful:
        print("Waiting for guidance images to be available...")
        # Give some time for the images to be generated
        import time
        time.sleep(10)
    
    # Use dictionary mapping for more efficient directory selection
    dataset_paths = {
        "train": (config.TRAIN_IMAGES_INPUT, config.TRAIN_GUIDANCE_DIR, config.TRAIN_IMAGES_GT),
        "test": (config.TEST_IMAGES_INPUT, config.TEST_GUIDANCE_DIR, config.TEST_IMAGES_GT),
        "val": (config.VAL_IMAGES_INPUT, config.VAL_GUIDANCE_DIR, config.VAL_IMAGES_GT)
    }
    
    if split not in dataset_paths:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")
    
    input_dir, guidance_dir, gt_dir = dataset_paths[split]
    
    # Create the dataset
    dataset = EdgeConnectDataset_G2(
        input_dir=input_dir,
        guidance_dir=guidance_dir,
        gt_dir=gt_dir if use_gt else None,
        image_size=config.IMAGE_SIZE,
        use_gt=use_gt
    )

    # Return the DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE_G2,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=2
    )


if __name__ == '__main__':

    train_loader = get_dataloader_g2(split="val")

    for batch in train_loader:
        input_img = batch["input_img"]
        guidance_img = batch["guidance_img"]
        mask = batch["mask"]
        gt_img = batch.get("gt_img", None)  # Optional ground truth image

        # Process your batch here
        print(f"Input shape: {input_img.shape}, Guidance shape: {guidance_img.shape}, Mask shape: {mask.shape}")
        if gt_img is not None:
            print(f"GT shape: {gt_img.shape}")
