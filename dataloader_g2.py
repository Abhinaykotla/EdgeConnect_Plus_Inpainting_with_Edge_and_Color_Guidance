# g2_dataloader.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import config

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


def get_dataloader_g2(input_dir, guidance_dir, gt_dir, batch_size=None, shuffle=True):
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
