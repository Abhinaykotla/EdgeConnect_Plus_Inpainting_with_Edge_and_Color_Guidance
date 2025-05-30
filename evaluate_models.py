import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips 
from config import config
from dataloader_g2 import get_dataloader_g2
from g2_model import InpaintingGeneratorG2
from loss_functions import VGG16FeatureExtractor, perceptual_loss, style_loss, l1_loss, calculate_fid, InceptionV3Features


def main():
    """
    Main evaluation function for EdgeConnect+ G2 model performance assessment.
    
    Evaluates the trained model against a test dataset using multiple image quality metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - L1 Loss
    - Perceptual Loss (VGG-based)
    - Style Loss
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - FID (Fréchet Inception Distance)
    
    Results are saved as visualizations and CSV metrics summaries.
    """
    device = config.DEVICE
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    save_images = True

    # Load VGG, LPIPS and Inception models for evaluation metrics
    print("INFO: Loading evaluation models...")
    vgg = VGG16FeatureExtractor().to(device).eval()
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    lpips_model.eval()
    inception = InceptionV3Features(device)  # Used for FID calculation

    # Load test data
    test_loader = get_dataloader_g2(split="val", batch_size=config.BATCH_SIZE_G2_INFERENCE)
    metrics_list = []  # Will hold per-image metrics

    # Load model
    model = InpaintingGeneratorG2().to(device)
    checkpoint_path = config.G2_MODEL_PATH
    print(f"INFO: Loading G2 model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["g2_state_dict"])
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)

    # Lists to store features for FID calculation
    # FID requires feature vectors from the Inception model
    real_features_list = []  # Ground truth image features
    fake_features_list = []  # Generated image features

    print("INFO: Processing images and collecting metrics...")
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            input_img = batch["input_img"].to(device)           # Shape: [B, 3, H, W]
            guidance_img = batch["guidance_img"].to(device)     # Shape: [B, 3, H, W]
            mask = batch["mask"].to(device)                     # Shape: [B, 1, H, W]
            gt_img = batch["gt_img"].to(device)                 # Shape: [B, 3, H, W]

            # Generate predictions with model
            pred_img = model(input_img, guidance_img, mask)     # Shape: [B, 3, H, W]
            pred_norm = (pred_img + 1) / 2  # Normalize from [-1,1] to [0,1] range
            gt_norm = gt_img  # GT is already in [0,1] range

            # Extract Inception features for FID calculation (done at batch level)
            real_features = inception.get_features(gt_norm)     # Shape: [B, feature_dim]
            fake_features = inception.get_features(pred_norm)   # Shape: [B, feature_dim]
            real_features_list.append(real_features)
            fake_features_list.append(fake_features)

            # Convert to numpy for per-image metrics
            # Change from [B, C, H, W] to [B, H, W, C] for image processing libraries
            pred_np = pred_norm.cpu().numpy().transpose(0, 2, 3, 1)
            gt_np = gt_norm.cpu().numpy().transpose(0, 2, 3, 1)

            # Calculate per-image metrics
            for i in range(pred_np.shape[0]):
                # Standard image quality metrics
                psnr = compare_psnr(gt_np[i], pred_np[i], data_range=1.0)
                ssim = compare_ssim(gt_np[i], pred_np[i], channel_axis=-1, data_range=1.0)
                
                # Loss-based metrics
                # Add unsqueeze to make single image into batch of 1 for loss functions
                l1 = l1_loss(pred_img[i:i+1], gt_img[i:i+1]).item()
                perc = perceptual_loss(vgg, pred_img[i:i+1], gt_img[i:i+1]).item()
                style = style_loss(vgg, pred_img[i:i+1], gt_img[i:i+1]).item()
                lpips_value = lpips_model(pred_img[i:i+1], gt_img[i:i+1]).item()

                # Store all metrics for this image
                metrics_list.append({
                    "index": batch_idx * config.BATCH_SIZE_G2_INFERENCE + i,
                    "psnr": psnr,
                    "ssim": ssim,
                    "l1_loss": l1,
                    "perceptual_loss": perc,
                    "style_loss": style,
                    "lpips": lpips_value
                })

                # Save sample images for visualization
                # Only save first two images from each batch to avoid too many outputs
                if save_images and i < 2:
                    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
                    
                    # Top row: Input, Guidance, Prediction
                    axs[0, 0].imshow(input_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[0, 0].set_title("Masked Input"); axs[0, 0].axis("off")

                    axs[0, 1].imshow(guidance_img[i].cpu().permute(1, 2, 0).numpy().clip(0, 1))
                    axs[0, 1].set_title("Guidance"); axs[0, 1].axis("off")

                    axs[0, 2].imshow(pred_np[i])
                    axs[0, 2].set_title("Prediction"); axs[0, 2].axis("off")

                    # Bottom row: Mask, Ground Truth, Difference Map
                    axs[1, 0].imshow(mask[i].cpu().squeeze(), cmap="gray")
                    axs[1, 0].set_title("Mask"); axs[1, 0].axis("off")

                    axs[1, 1].imshow(gt_np[i])
                    axs[1, 1].set_title("Ground Truth"); axs[1, 1].axis("off")

                    # Absolute difference shows error regions clearly
                    axs[1, 2].imshow(np.abs(gt_np[i] - pred_np[i]))
                    axs[1, 2].set_title("Difference"); axs[1, 2].axis("off")

                    plt.tight_layout()
                    fname = os.path.join(config.EVAL_RESULTS_DIR, f"sample_{batch_idx}_{i}.png")
                    plt.savefig(fname)
                    plt.close()

    # Calculate FID score using all collected features
    # FID measures distance between feature distributions of real and generated images
    print("INFO: Calculating FID score...")
    all_real_features = torch.cat(real_features_list, dim=0)
    all_fake_features = torch.cat(fake_features_list, dim=0)
    fid_score = calculate_fid(all_real_features, all_fake_features)

    # Create DataFrame from per-image metrics for analysis and storage
    df = pd.DataFrame(metrics_list)

    # Add global metrics summary row with FID
    # This creates an aggregate view across the entire test set
    global_metrics = {
        "index": "GLOBAL",
        "psnr": df["psnr"].mean(),
        "ssim": df["ssim"].mean(),
        "l1_loss": df["l1_loss"].mean(),
        "perceptual_loss": df["perceptual_loss"].mean(),
        "style_loss": df["style_loss"].mean(),
        "lpips": df["lpips"].mean(),
        "fid": fid_score  # FID is a dataset-level metric, not per-image
    }

    # Append global metrics to the dataframe
    global_df = pd.DataFrame([global_metrics])
    df = pd.concat([df, global_df], ignore_index=True)

    # Save all metrics to CSV
    csv_path = os.path.join(config.EVAL_RESULTS_DIR, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"INFO: All metrics saved to: {csv_path}")

    # Print summary statistics
    print("\nRESULT: Evaluation Summary:")
    metrics_summary = df.iloc[:-1].describe().loc[["mean", "50%"]].rename(index={"50%": "median"})
    print(metrics_summary)
    print(f"FID Score: {fid_score:.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required for Windows to properly handle multiprocessing
    main()