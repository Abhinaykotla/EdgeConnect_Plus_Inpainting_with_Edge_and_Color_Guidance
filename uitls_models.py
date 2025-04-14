
import os
import cv2
import torch
import numpy as np
from config import config
from utils_dl import apply_canny, dilate_mask, remove_mask_edge
from g1_model import EdgeGenerator

def generate_edges_with_g1(model_path, input_path, save_dir=None):
    """
    Load a trained G1 model and generate edge maps for one image or all images in a folder.

    Args:
        model_path (str): Path to the G1 model checkpoint (.pth file).
        input_path (str): Path to a single masked image or a directory of masked images.
        save_dir (str): Optional directory to save the predicted edge maps.

    Returns:
        List of predicted edge tensors.
    """
    # Load trained G1 model
    g1 = EdgeGenerator().to(config.DEVICE)
    state_dict = torch.load(model_path, map_location=config.DEVICE)["g1_state_dict"]
    g1.load_state_dict(state_dict)
    g1.eval()

    # Make sure save_dir exists if saving
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Prepare image paths
    if os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".jpg")]
    else:
        image_paths = [input_path]

    pred_edges_list = []

    with torch.no_grad():
        for img_path in image_paths:
            # Load masked image
            input_img = cv2.imread(img_path)
            if input_img is None:
                print(f"❌ Could not read image: {img_path}")
                continue

            input_img_resized = cv2.resize(input_img, (config.IMAGE_SIZE, config.IMAGE_SIZE))

            # Mask extraction: white pixels are missing
            mask_binary = np.all(input_img_resized > 245, axis=-1).astype(np.float32)
            raw_mask = 255 - mask_binary * 255
            dilated_mask_np = dilate_mask(raw_mask)

            # Canny edges
            input_edge = apply_canny(input_img_resized)
            input_edge = remove_mask_edge(dilated_mask_np, input_edge)

            # Grayscale input
            gray = cv2.cvtColor(input_img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            # Convert to tensors
            input_edge_tensor = torch.from_numpy(input_edge).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE)
            mask_tensor = torch.from_numpy(1.0 - dilated_mask_np).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE)
            gray_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(config.DEVICE)

            # Run G1
            pred_edge = g1(input_edge_tensor, mask_tensor, gray_tensor)
            pred_edges_list.append(pred_edge.squeeze(0).cpu())  # [1, 1, H, W] → [1, H, W]

            # Save edge image if requested
            if save_dir:
                edge_np = (pred_edge.squeeze().cpu().numpy() * 255).astype(np.uint8)
                save_path = os.path.join(save_dir, os.path.basename(img_path))
                cv2.imwrite(save_path, edge_np)

    print(f"✅ Processed {len(pred_edges_list)} image(s)")
    return pred_edges_list
