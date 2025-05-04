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
    device = config.DEVICE
    
    # Create necessary directories
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.DEMO_RESULTS_DIR, exist_ok=True)  # Create demo results directory

    print(f"Results will be saved to: {config.DEMO_RESULTS_DIR}")

    # Load model
    model = InpaintingGeneratorG2().to(device)
    checkpoint_path = config.G2_MODEL_PATH
    print(f"✅ Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["g2_state_dict"])
    model.eval()

    # Check if demo directories exist and have files
    demo_input_dir = config.DEMO_IMAGES_INPUT
    if not os.path.exists(demo_input_dir):
        print(f"Error: Demo input directory '{demo_input_dir}' does not exist.")
        print("Please create this directory and add your demo images.")
        sys.exit(1)
        
    if len([f for f in os.listdir(demo_input_dir) if f.endswith('.jpg')]) == 0:
        print(f"Error: No jpg images found in demo input directory '{demo_input_dir}'.")
        print("Please add your demo images to this directory.")
        sys.exit(1)

    # Load demo data
    try:
        demo_loader = get_dataloader_g2(split="demo", batch_size=config.BATCH_SIZE_G2_INFERENCE, use_gt=True)
    except ValueError as e:
        print(f"Error loading demo dataset: {e}")
        print("Ensure your demo directories contain valid images.")
        sys.exit(1)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(demo_loader, desc="Processing demo images")):
            input_img = batch["input_img"].to(device)
            guidance_img = batch["guidance_img"].to(device)
            mask = batch["mask"].to(device)
            
            # Generate inpainted images
            pred_img = model(input_img, guidance_img, mask)
            pred_norm = (pred_img + 1) / 2
            
            # Convert to numpy for visualization
            pred_np = pred_norm.cpu().numpy().transpose(0, 2, 3, 1)
            
            # Handle ground truth if available
            has_gt = "gt_img" in batch
            gt_np = None
            if has_gt:
                gt_img = batch["gt_img"].to(device)
                gt_norm = gt_img
                gt_np = gt_norm.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(pred_np.shape[0]):
                if has_gt:
                    # Plot with ground truth
                    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
                    
                    axs[0, 0].imshow(input_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[0, 0].set_title("Masked Input"); axs[0, 0].axis("off")

                    axs[0, 1].imshow(guidance_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[0, 1].set_title("Guidance"); axs[0, 1].axis("off")

                    axs[0, 2].imshow(pred_np[i])
                    axs[0, 2].set_title("Prediction"); axs[0, 2].axis("off")

                    axs[1, 0].imshow(mask[i].cpu().squeeze(), cmap="gray")
                    axs[1, 0].set_title("Mask"); axs[1, 0].axis("off")

                    axs[1, 1].imshow(gt_np[i])
                    axs[1, 1].set_title("Ground Truth"); axs[1, 1].axis("off")

                    axs[1, 2].imshow(np.abs(gt_np[i] - pred_np[i]))
                    axs[1, 2].set_title("Difference"); axs[1, 2].axis("off")
                else:
                    # Plot without ground truth
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
                fname = os.path.join(config.DEMO_RESULTS_DIR, f"demo_sample_{batch_idx}_{i}.png")
                plt.savefig(fname)
                plt.close()
                
                # Also save the inpainted image separately
                plt.figure(figsize=(8, 8))
                plt.imshow(pred_np[i])
                plt.axis('off')
                plt.tight_layout()
                result_fname = os.path.join(config.DEMO_RESULTS_DIR, f"result_{batch_idx}_{i}.png")
                plt.savefig(result_fname)
                plt.close()

    print(f"✅ Demo complete! Results saved to {config.DEMO_RESULTS_DIR}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()