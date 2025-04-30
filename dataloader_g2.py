# g2_dataloader.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config
from utils_dl import apply_canny, dilate_mask, gen_raw_mask
from dataloader_g1 import get_dataloader_g1

def gen_gidance_img(input_img, guidance_img):
    pass

def validate_edge_map(split="train"):
    """
    Validates if the number of images in the input folder matches the number of images in the edge folder.
    If the number of files doesn't match or any edge map is missing, the edge folder is cleared, and edge maps
    are regenerated using the _generate_edge_maps function.

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
        mask = batch["mask"].to(config.DEVICE)              # Shape: (batch_size, 1, H, W)

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


class EdgeConnectDataset_G2(Dataset):
    def __init__(self, input_dir, guidance_dir, gt_dir, image_size=256):
        self.input_dir = input_dir
        self.guidance_dir = guidance_dir
        self.gt_dir = gt_dir
        self.image_size = image_size

        self.input_files = sorted([f.name for f in os.scandir(input_dir) if f.name.endswith('.jpg')])
        self.guidance_files = sorted([f.name for f in os.scandir(guidance_dir) if f.name.endswith('.jpg')])
        self.gt_files = sorted([f.name for f in os.scandir(gt_dir) if f.name.endswith('.jpg')])

        assert len(self.input_files) == len(self.guidance_files) == len(self.gt_files), \
            "Mismatch in number of images."

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0,1] and permutes to [C,H,W]
            transforms.Resize((self.image_size, self.image_size))
        ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        guidance_path = os.path.join(self.guidance_dir, self.guidance_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        # Read RGB images
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        guidance_img = cv2.cvtColor(cv2.imread(guidance_path), cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)

        # Mask: white (255,255,255) → missing, others → known
        mask = np.all(input_img > 245, axis=-1).astype(np.float32)  # shape: (H,W)
        mask = (1.0 - mask).astype(np.float32)  # 1.0 = missing, 0.0 = known

        # Apply transforms
        input_tensor = self.transform(input_img)
        guidance_tensor = self.transform(guidance_img)
        gt_tensor = self.transform(gt_img)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        return {
            "input_img": input_tensor,
            "guidance_img": guidance_tensor,
            "mask": mask_tensor,
            "gt_img": gt_tensor
        }


def get_dataloader_g2(input_dir, guidance_dir, gt_dir, batch_size=config.BATCH_SIZE, shuffle=True):
    dataset = EdgeConnectDataset_G2(
        input_dir=input_dir,
        guidance_dir=guidance_dir,
        gt_dir=gt_dir,
        image_size=config.IMAGE_SIZE
    )

    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
