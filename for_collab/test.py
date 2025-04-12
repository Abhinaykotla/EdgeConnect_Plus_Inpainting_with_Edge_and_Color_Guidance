import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

from g1_model import EdgeGenerator
from config import config
from dataloader import apply_canny, remove_mask_edge, dilate_mask

def load_specific_checkpoint(model, checkpoint_path):
    """Load a specific checkpoint for the model."""
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['g1_state_dict'])
    if 'ema_shadow' in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint['ema_shadow']:
                param.data.copy_(checkpoint['ema_shadow'][name])
        print("✅ EMA shadow exists in the checkpoint and has been loaded.")
    else:
        print("❌ EMA shadow does not exist in the checkpoint.")
    print(f"✅ Loaded checkpoint from {checkpoint_path}")

def display_images(input_image, mask, gray_image, pred_edge):
    """Display all images using matplotlib"""
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.title('Input Image')
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(142)
    plt.title('Mask')
    plt.imshow(mask.cpu().squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(143)
    plt.title('Grayscale Image')
    plt.imshow(gray_image.cpu().squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(144)
    plt.title('Generated Edge')
    plt.imshow(pred_edge.cpu().squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def preprocess_image(image_path):
    """Preprocess the custom image for the model."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    mask_binary = np.all(image > 245, axis=-1).astype(np.float32)
    raw_mask = 255 - mask_binary * 255
    dilated_mask_np = dilate_mask(raw_mask)

    input_edge = apply_canny(image)
    input_edge = np.where(dilated_mask_np > 0.5, 1.0, input_edge)

    input_edge = torch.from_numpy(input_edge).float().unsqueeze(0)
    mask_for_model = 1.0 - dilated_mask_np
    mask_for_model = torch.from_numpy(mask_for_model).float().unsqueeze(0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    gray = torch.from_numpy(gray).float().unsqueeze(0)

    return input_edge, mask_for_model, gray, image

def test_custom_image(image_path, save_output_path=None):
    """Test the EdgeGenerator model on a custom image and optionally save the predicted edge."""
    print("\n🔹 Initializing Model & Loading Checkpoint...\n")
    g1 = EdgeGenerator().to(config.DEVICE)
    checkpoint_path = os.path.join("/content/drive/MyDrive/edgeconnect/models/checkpoints", "checkpoint_epoch_43.pth")
    load_specific_checkpoint(g1, checkpoint_path)
    g1.eval()

    input_edge, mask, gray, original_image = preprocess_image(image_path)
    input_edge = input_edge.to(config.DEVICE).unsqueeze(0)
    mask = mask.to(config.DEVICE).unsqueeze(0)
    gray = gray.to(config.DEVICE).unsqueeze(0)

    print(f"🔹 Running Inference on Custom Image...\n")
    with torch.no_grad():
        pred_edge = g1(input_edge, mask, gray)

        display_images(original_image, mask, gray, pred_edge[0])

        # Save the predicted edge
        if save_output_path:
            pred_np = pred_edge[0].cpu().squeeze().numpy() * 255
            pred_np = pred_np.astype(np.uint8)
            Image.fromarray(pred_np).save(save_output_path)
            print(f"💾 Saved predicted edge to: {save_output_path}")

    print("\n✅ Testing Completed.\n")

if __name__ == '__main__':
    # Set custom input image path and save path
    custom_image_path = "/content/drive/MyDrive/edgeconnect/data_archive/CelebA/test_input/000024.jpg"
    output_image_path = "/content/drive/MyDrive/edgeconnect/results/output_edge_000024.png"
    test_custom_image(custom_image_path, save_output_path=output_image_path)
