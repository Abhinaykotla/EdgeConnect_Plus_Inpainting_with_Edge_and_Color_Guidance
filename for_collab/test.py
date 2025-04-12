import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from g1_model import EdgeGenerator
from config import config
from dataloader import apply_canny, remove_mask_egde, get_dataloader_g1

def load_specific_checkpoint(model, checkpoint_path):
    """
    Load a specific checkpoint for the model.
    """
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

def plot_and_display_images(input_image, pred_edge):
    """
    Plot and display input images and generated edges side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot input image
    axes[0].imshow(input_image.cpu().squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Plot generated edge
    axes[1].imshow(pred_edge.cpu().squeeze(), cmap='gray')
    axes[1].set_title('Generated Edge')
    axes[1].axis('off')

    # Display the plot
    plt.show()

def plot_and_save_images(input_image, pred_edge, output_dir, image_name):
    """
    Plot and save input images and generated edges side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot input image
    axes[0].imshow(input_image.cpu().squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Plot generated edge
    axes[1].imshow(pred_edge.cpu().squeeze(), cmap='gray')
    axes[1].set_title('Generated Edge')
    axes[1].axis('off')

    # Save the plot
    save_path = os.path.join(output_dir, f"plot_{image_name}.png")
    plt.savefig(save_path)
    plt.close(fig)

def preprocess_image(image_path):
    """
    Preprocess the custom image for the model.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    
    # Extract mask: Consider pixels as missing if all RGB values > 245
    mask_binary = np.all(image > 245, axis=-1).astype(np.float32)  # Shape: (H, W)
    mask = 255 - mask_binary * 255  # Invert mask (0s for missing pixels, 255s for known pixels)
    mask = torch.from_numpy(mask).unsqueeze(0)  # Shape: (1, H, W)

    # Generate Edge Maps
    input_edge = apply_canny(image)  # Edges from masked image
    
    # Apply mask edge removal
    input_edge = remove_mask_egde(mask, input_edge)

    # Convert to Tensor if not already
    if not isinstance(input_edge, torch.Tensor):
        input_edge = torch.from_numpy(input_edge).float().unsqueeze(0)  # Shape: (1, H, W)
    
    mask = mask / 255.0  # Normalize mask to [0, 1]

    # Concatenate input_edge and mask along the channel dimension
    input_combined = torch.cat((input_edge, mask), dim=0)  # Shape: (2, H, W)

    return input_combined.unsqueeze(0)  # Add batch dimension

def test_custom_image(image_path=None):
    """
    Test the EdgeGenerator (G1) model on a custom image or test dataset.
    """
    print("\n🔹 Initializing Model & Loading Checkpoint...\n")

    # Initialize the generator model
    g1 = EdgeGenerator().to(config.DEVICE)

    # Load the specific checkpoint
    checkpoint_path = os.path.join("/content/drive/MyDrive/edgeconnect/", "checkpoint_epoch_3.pth")
    load_specific_checkpoint(g1, checkpoint_path)

    # Set the model to evaluation mode
    g1.eval()

    if image_path:
        # Preprocess the custom image
        input_combined = preprocess_image(image_path).to(config.DEVICE)

        print(f"🔹 Running Inference on Custom Image...\n")

        with torch.no_grad():
            # Generate edges using the trained generator model
            pred_edge = g1(input_combined[:, 0:1, :, :], input_combined[:, 1:2, :, :])

            # Plot and display the input image and generated edge
            plot_and_display_images(input_combined[0, 0], pred_edge)

    else:
        print("🔹 No custom image provided. Using test dataset...\n")

        # Load the test dataset
        test_dataloader = get_dataloader_g1(split="test", use_mask=True)

        # Directory to save generated images
        output_dir = os.path.join(config.MODEL_CHECKPOINT_DIR, "test_results")
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                input_edges, mask = (
                    batch["input_edge"].to(config.DEVICE),   
                    batch["mask"].to(config.DEVICE)
                )

                # Generate edges using the trained generator model
                pred_edge = g1(input_edges, mask)

                # Plot and save the input image and generated edge
                for i in range(input_edges.size(0)):
                    image_name = f"batch{batch_idx}_img{i}"
                    plot_and_save_images(input_edges[i], pred_edge[i], output_dir, image_name)

                print(f"  🔹 Batch [{batch_idx+1}/{len(test_dataloader)}] - Plots saved.")

    print("\n✅ Testing Completed.\n")

if __name__ == '__main__':
    # Path to the custom image (set to None to use test dataset)
    custom_image_path = "/content/drive/MyDrive/edgeconnect/data_archive/CelebA/test_input/000007.jpg"  # "/path/to/your/custom_image.jpg" or None
    test_custom_image(custom_image_path)