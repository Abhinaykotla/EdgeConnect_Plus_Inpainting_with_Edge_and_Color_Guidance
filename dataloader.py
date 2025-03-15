import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config  # Import updated config file

def apply_canny(image):
    """
    Apply Canny edge detection to an image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(image, 50, 150)
    edges = (255 - edges).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return edges

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

        self.input_files = sorted(os.listdir(input_dir))
        self.gt_files = sorted(os.listdir(gt_dir))
        
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        # Load Images
        input_img = cv2.imread(input_path)  # Masked Image
        gt_img = cv2.imread(gt_path)  # Ground Truth Image

        # Resize Images
        input_img = cv2.resize(input_img, (self.image_size, self.image_size))
        gt_img = cv2.resize(gt_img, (self.image_size, self.image_size))

        # Extract mask: Consider pixels as missing if all RGB values > 240
        mask = np.all(input_img > 245, axis=-1).astype(np.float32) # Shape: (H, W)
        mask = 255 - mask * 255  # Invert mask (0s for missing pixels, 255s for known pixels)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Shape: (1, H, W)

        # Create dilated mask for edge detection
        # First convert back to numpy for OpenCV operations
        mask_np = mask.squeeze(0).numpy()
        dilated_mask = cv2.dilate(mask_np, np.ones((5, 5), np.uint8), iterations=1)  # Dilate mask for inpainting

        print("Dilated Mask Shape:", dilated_mask.shape)
        print("min and max values:", dilated_mask.min(), dilated_mask.max())

        # Generate Edge Maps
        input_edge = apply_canny(input_img)  # Edges from masked image
        
        # Multiply with dilated mask (both are numpy arrays)
        input_edge = input_edge * dilated_mask  # Zero out edges outside mask

        gt_edge = apply_canny(gt_img)  # Edges from ground truth image

        # Convert to Tensors if not already
        if not isinstance(input_edge, torch.Tensor):
            input_edge = torch.from_numpy(input_edge).unsqueeze(0)  # Shape: (1, H, W)
        
        if not isinstance(gt_edge, torch.Tensor):
            gt_edge = torch.from_numpy(gt_edge).unsqueeze(0)  # Shape: (1, H, W)

        # Return mask if enabled
        if self.use_mask:
            return {"input_edge": input_edge, "gt_edge": gt_edge, "mask": mask}

        return {"input_edge": input_edge, "gt_edge": gt_edge}

# Initialize DataLoader for G1 with optional mask input
def get_dataloader_g1(split="train", use_mask=False):
    if split == "train":
        dataset = EdgeConnectDataset_G1(config.TRAIN_IMAGES_INPUT, config.TRAIN_IMAGES_GT, config.IMAGE_SIZE, use_mask)
    elif split == "test":
        dataset = EdgeConnectDataset_G1(config.TEST_IMAGES_INPUT, config.TEST_IMAGES_GT, config.IMAGE_SIZE, use_mask)
    elif split == "val":
        dataset = EdgeConnectDataset_G1(config.VAL_IMAGES_INPUT, config.VAL_IMAGES_GT, config.IMAGE_SIZE, use_mask)
    else:
        raise ValueError("Invalid dataset split. Choose from 'train', 'test', or 'val'.")

    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)


if __name__ == "__main__":
    # Test DataLoader
    dataloader = get_dataloader_g1(split="val", use_mask=True)
    for batch in dataloader:
        print(batch["input_edge"].shape, batch["gt_edge"].shape, batch["mask"].shape)
        break