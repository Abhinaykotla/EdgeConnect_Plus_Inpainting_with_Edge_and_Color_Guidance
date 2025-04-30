# g2_dataloader.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config
from dataloader_g1 import get_dataloader_g1
from utils_dl import apply_canny, dilate_mask, gen_raw_mask
from functools import lru_cache
from pathlib import Path


def gen_gidance_img(input_img, guidance_img):
    pass


def validate_edge_map(split="train"):
    """
    Validates if the number of images in the input folder matches the number of images in the edge folder.
    If the number of files doesn't match or any edge map is missing, the edge folder is cleared, and edge maps are regenerated using the _generate_edge_maps function.

    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').

    Returns:
        bool: True if validation and regeneration (if needed) are successful, False otherwise.
    """
    # Select input and edge directories based on the split
    if split == "train":
        input_dir = config.TRAIN_IMAGES_INPUT
        edge_dir = config.TRAIN_EDGE_DIR
    elif split == "test":
        input_dir = config.TEST_IMAGES_INPUT
        edge_dir = config.TEST_EDGE_DIR
    elif split == "val":
        input_dir = config.VAL_IMAGES_INPUT
        edge_dir = config.VAL_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")

    input_files = sorted([f.name for f in os.scandir(input_dir) if f.name.endswith('.jpg')])
    edge_files = sorted([f.name for f in os.scandir(edge_dir) if f.name.endswith('.jpg')])

    # Check if the number of files matches
    if len(input_files) != len(edge_files):
        print(f"Mismatch in number of images: {len(input_files)} input images vs {len(edge_files)} edge images.")
        print("Clearing edge folder and regenerating edge maps...")
        _clear_folder(edge_dir)
        _generate_edge_maps(split=split, batch_size=config.BATCH_SIZE)
        return False

    # Check if all input images have corresponding edge maps
    for input_file in input_files:
        expected_edge_file = f"{os.path.splitext(input_file)[0]}_edgemap.jpg"
        if expected_edge_file not in edge_files:
            print(f"Missing edge map for {input_file}. Clearing edge folder and regenerating edge maps...")
            _clear_folder(edge_dir)
            _generate_edge_maps(split=split, batch_size=config.BATCH_SIZE)
            return False

    print("Number of images and corresponding edge maps match.")
    return True


def validate_guidance_images(split="train"):
    """
    Validates if guidance images exist for all input images.
    If any guidance image is missing, they will be generated.

    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').

    Returns:
        bool: True if validation was successful, False if regeneration was needed.
    """
    # Select directories based on the split
    if split == "train":
        input_dir = config.TRAIN_IMAGES_INPUT
        guidance_dir = config.TRAIN_GUIDANCE_DIR
    elif split == "test":
        input_dir = config.TEST_IMAGES_INPUT
        guidance_dir = config.TEST_GUIDANCE_DIR
    elif split == "val":
        input_dir = config.VAL_IMAGES_INPUT
        guidance_dir = config.VAL_GUIDANCE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")

    # Ensure guidance directory exists
    os.makedirs(guidance_dir, exist_ok=True)
    
    # Get list of input and guidance images
    input_path = Path(input_dir)
    guidance_path = Path(guidance_dir)
    input_files = sorted([f.name for f in input_path.glob("*.jpg")])
    guidance_files = sorted([f.name for f in guidance_path.glob("*.jpg")])
    
    # Check if numbers match
    if len(input_files) != len(guidance_files):
        print(f"Mismatch in number of images: {len(input_files)} input images vs {len(guidance_files)} guidance images.")
        print("Generating missing guidance images...")
        _generate_guidance_images(split=split)
        return False
    
    # Check if each input has a corresponding guidance image
    for input_file in input_files:
        basename = os.path.splitext(input_file)[0]
        expected_guidance_file = f"{basename}.jpg"
        if expected_guidance_file not in guidance_files:
            print(f"Missing guidance image for {input_file}. Generating guidance images...")
            _generate_guidance_images(split=split)
            return False
    
    print("All guidance images exist.")
    return True


def _clear_folder(folder_path):
    """
    Clears all files in the specified folder.

    Args:
        folder_path (str): Path to the folder to clear.
    """
    for file in os.scandir(folder_path):
        os.remove(file.path)
    print(f"Cleared folder: {folder_path}")


def _generate_edge_maps(split="train", batch_size=32):
    """
    Generates edge maps for all input images in batches and saves them in the edge folder.

    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').
        batch_size (int): Number of images to process in a single batch.
    """
    # Select input and edge directories based on the split
    if split == "train":
        edge_dir = config.TRAIN_EDGE_DIR
    elif split == "test":
        edge_dir = config.TEST_EDGE_DIR
    elif split == "val":
        edge_dir = config.VAL_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")

    # Ensure the edge directory exists
    os.makedirs(edge_dir, exist_ok=True)

    # Load the checkpoint
    checkpoint_path = config.G1_MODEL_PATH  # Path to the G1 model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    # Initialize the model architecture
    from g1_model import EdgeGenerator  # Replace with your actual model class
    model = EdgeGenerator()  # Initialize the model

    # Check if the checkpoint contains a full dictionary or just the state_dict
    if "g1_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["g1_state_dict"])  # Load only the model weights
    else:
        model.load_state_dict(checkpoint)  # Directly load raw weights if no wrapper exists

    model.eval()  # Set the model to evaluation mode
    model.to(config.DEVICE)  # Move the model to the specified device (e.g., GPU)

    # Initialize the dataloader with use_gt=False
    dataloader = get_dataloader_g1(split=split, use_mask=True, use_gt=False)

    # Process images in batches
    for batch_idx, batch in enumerate(dataloader):
        input_edge = batch["input_edge"].to(config.DEVICE)  # Shape: (batch_size, 1, H, W)
        gray = batch["gray"].to(config.DEVICE)              # Shape: (batch_size, 1, H, W)
        mask = batch["mask"].to(config.DEVICE)              # Shape: (1, H, W)

        # Generate edge maps for the batch
        with torch.no_grad():
            edge_maps = model(input_edge, mask, gray)  # Pass the batch through the model

        # Save the generated edge maps
        for j in range(edge_maps.size(0)):
            edge_map = edge_maps[j].cpu().numpy().squeeze()  # Convert to numpy and remove channel dimension
            edge_map = (edge_map * 255).astype(np.uint8)     # Scale to [0, 255]
            edge_path = os.path.join(edge_dir, f"edge_map_{batch_idx * batch_size + j}.jpg")
            cv2.imwrite(edge_path, edge_map)

        print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    print(f"Generated edge maps for all input images in {edge_dir}.")


def _generate_guidance_images(split="train"):
    """
    Generates guidance images for all input images based on edge maps.
    
    Args:
        split (str): Dataset split to use ('train', 'test', or 'val').
    """
    # Select directories based on the split
    if split == "train":
        input_dir = config.TRAIN_IMAGES_INPUT
        guidance_dir = config.TRAIN_GUIDANCE_DIR
        edge_dir = config.TRAIN_EDGE_DIR
    elif split == "test":
        input_dir = config.TEST_IMAGES_INPUT
        guidance_dir = config.TEST_GUIDANCE_DIR
        edge_dir = config.TEST_EDGE_DIR
    elif split == "val":
        input_dir = config.VAL_IMAGES_INPUT
        guidance_dir = config.VAL_GUIDANCE_DIR
        edge_dir = config.VAL_EDGE_DIR
    else:
        raise ValueError("Invalid split. Choose from 'train', 'test', or 'val'.")
    
    # First make sure edge maps exist
    validate_edge_map(split)
    
    # Ensure guidance directory exists
    os.makedirs(guidance_dir, exist_ok=True)
    
    # Get all input images
    input_files = sorted([f for f in os.scandir(input_dir) if f.name.endswith('.jpg')])
    
    print(f"Generating guidance images for {len(input_files)} input images...")
    
    # Process each input image
    for file in input_files:
        # Get corresponding edge map
        basename = os.path.splitext(file.name)[0]
        input_path = file.path
        edge_path = os.path.join(edge_dir, f"edge_map_{basename.split('_')[-1]}.jpg")
        guidance_path = os.path.join(guidance_dir, file.name)
        
        # Read images
        input_img = cv2.imread(input_path)
        edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        
        if input_img is None or edge_img is None:
            print(f"Warning: Could not read input or edge image for {file.name}")
            continue
        
        # Generate guidance image using input and edge
        # This would typically call the gen_guidance_img function
        # For now, we'll use a simple combination as a placeholder
        guidance_img = gen_gidance_img(input_img, edge_img)
        
        # Save the guidance image
        cv2.imwrite(guidance_path, guidance_img)
    
    print(f"Generated guidance images in {guidance_dir}")


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


def get_dataloader_g2(split="train", batch_size=config.BATCH_SIZE, shuffle=True, use_gt=True):
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
    # First, validate edge maps and guidance images
    validate_edge_map(split)
    validate_guidance_images(split)
    
    # Use dictionary mapping for more efficient directory selection
    dataset_paths = {
        "train": (config.TRAIN_IMAGES_INPUT, config.TRAIN_GUIDANCE_DIR, config.TRAIN_GT_DIR),
        "test": (config.TEST_IMAGES_INPUT, config.TEST_GUIDANCE_DIR, config.TEST_GT_DIR),
        "val": (config.VAL_IMAGES_INPUT, config.VAL_GUIDANCE_DIR, config.VAL_GT_DIR)
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
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=2
    )
