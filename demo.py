# Demo script for EdgeConnect+ that demonstrates the inpainting capabilities of the G2 model
# Loads a pre-trained model and performs inference on demo images, visualizing and saving results

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from config import config
from dataloader_g2 import get_dataloader_g2
from g2_model import InpaintingGeneratorG2


def main():
    """
    Main function to demonstrate EdgeConnect+ inpainting on demo images.
    
    This function:
    1. Sets up the environment and necessary directories
    2. Loads the pre-trained G2 inpainting model
    3. Processes demo images and generates inpainted results
    4. Creates visualizations comparing input, guidance, and output
    5. Saves results to the configured demo directory
    
    No parameters or return values as this is the entry point.
    """
    device = config.DEVICE
    
    # Create necessary directories
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.DEMO_RESULTS_DIR, exist_ok=True)  # Create demo results directory

    print(f"INFO: Results will be saved to: {config.DEMO_RESULTS_DIR}")

    # Check if demo directories exist and have files
    demo_input_dir = config.DEMO_IMAGES_INPUT
    if not os.path.exists(demo_input_dir):
        print(f"ERROR: Demo input directory '{demo_input_dir}' does not exist.")
        print("Please create this directory and add your demo images.")
        sys.exit(1)
        
    if len([f for f in os.listdir(demo_input_dir) if f.endswith('.jpg')]) == 0:
        print(f"ERROR: No jpg images found in demo input directory '{demo_input_dir}'.")
        print("Please add your demo images to this directory.")
        sys.exit(1)

    # Load demo data
    try:
        # Load images using a smaller batch size for inference to reduce memory usage
        demo_loader = get_dataloader_g2(split="demo", batch_size=config.BATCH_SIZE_G2_INFERENCE, use_gt=True)
    except ValueError as e:
        print(f"ERROR: Loading demo dataset: {e}")
        print("Ensure your demo directories contain valid images.")
        sys.exit(1)


    # Load model
    model = InpaintingGeneratorG2().to(device)
    checkpoint_path = config.G2_MODEL_PATH
    print(f"INFO: Loading G2 model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["g2_state_dict"])
    model.eval()  # Set model to evaluation mode to disable dropout, batch norm updates, etc.

    # Disable gradient calculation for inference to save memory and speed up computation
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(demo_loader, desc="Processing demo images")):
            # Move tensors to the device (CPU/GPU)
            input_img = batch["input_img"].to(device)      # Shape: [B, 3, H, W]
            guidance_img = batch["guidance_img"].to(device)  # Shape: [B, 3, H, W]
            mask = batch["mask"].to(device)                # Shape: [B, 1, H, W]
            
            # Generate inpainted images
            pred_img = model(input_img, guidance_img, mask)  # Shape: [B, 3, H, W]
            # Normalize predictions from [-1,1] to [0,1] range for visualization
            pred_norm = (pred_img + 1) / 2
            
            # Convert tensors to numpy arrays and rearrange dimensions for plotting
            # From [B, C, H, W] to [B, H, W, C] format
            pred_np = pred_norm.cpu().numpy().transpose(0, 2, 3, 1)
            
            # Check if ground truth is available in the batch
            has_gt = "gt_img" in batch
            gt_np = None
            if has_gt:
                gt_img = batch["gt_img"].to(device)
                gt_norm = gt_img  # GT is already in [0,1] range
                gt_np = gt_norm.cpu().numpy().transpose(0, 2, 3, 1)

            # Process each image in the batch individually
            for i in range(pred_np.shape[0]):
                if has_gt:
                    # Create visualization with ground truth comparison (2x3 grid)
                    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
                    
                    # Input image
                    axs[0, 0].imshow(input_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[0, 0].set_title("Masked Input"); axs[0, 0].axis("off")

                    # Guidance image (edge and color guidance)
                    axs[0, 1].imshow(guidance_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[0, 1].set_title("Guidance"); axs[0, 1].axis("off")

                    # Inpainted result
                    axs[0, 2].imshow(pred_np[i])
                    axs[0, 2].set_title("Prediction"); axs[0, 2].axis("off")

                    # Mask (1=valid, 0=hole)
                    axs[1, 0].imshow(mask[i].cpu().squeeze(), cmap="gray")
                    axs[1, 0].set_title("Mask"); axs[1, 0].axis("off")

                    # Ground truth image
                    axs[1, 1].imshow(gt_np[i])
                    axs[1, 1].set_title("Ground Truth"); axs[1, 1].axis("off")

                    # Absolute difference between prediction and ground truth
                    axs[1, 2].imshow(np.abs(gt_np[i] - pred_np[i]))
                    axs[1, 2].set_title("Difference"); axs[1, 2].axis("off")
                else:
                    # Create visualization without ground truth (1x4 grid)
                    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                    
                    axs[0].imshow(input_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[0].set_title("Masked Input"); axs[0].axis("off")

                    axs[1].imshow(guidance_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[1].set_title("Guidance"); axs[1].axis("off")
                    
                    axs[2].imshow(mask[i].cpu().squeeze(), cmap="gray")
                    axs[2].set_title("Mask"); axs[2].axis("off")

                    axs[3].imshow(pred_np[i])
                    axs[3].set_title("Prediction"); axs[3].axis("off")

                plt.tight_layout()
                # Save the full comparison visualization
                fname = os.path.join(config.DEMO_RESULTS_DIR, f"demo_sample_{batch_idx}_{i}.png")
                plt.savefig(fname)
                plt.close()
                
                # Also save the inpainted image separately for easier viewing
                plt.figure(figsize=(8, 8))
                plt.imshow(pred_np[i])
                plt.axis('off')
                plt.tight_layout()
                result_fname = os.path.join(config.DEMO_RESULTS_DIR, f"result_{batch_idx}_{i}.png")
                plt.savefig(result_fname)
                plt.close()

    print(f"INFO: Demo complete! Results saved to {config.DEMO_RESULTS_DIR}")


if __name__ == "__main__":
    # Support for freezing applications that use multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()