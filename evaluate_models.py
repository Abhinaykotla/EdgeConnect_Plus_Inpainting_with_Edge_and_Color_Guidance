
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from config import config
from dataloader_g2 import get_dataloader_g2
from g2_model import InpaintingGeneratorG2
from loss_functions import VGG16FeatureExtractor, perceptual_loss, style_loss, l1_loss

# Set device
device = config.DEVICE

# Output directory
os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
save_images = True  # Toggle this to save visuals

# Load model
model = InpaintingGeneratorG2().to(device)
checkpoint_files = sorted(Path(config.MODEL_CHECKPOINT_DIR_G2).glob("checkpoint_epoch_*.pth"))
if not checkpoint_files:
    raise FileNotFoundError("No G2 checkpoint found.")
checkpoint_path = checkpoint_files[-1]
print(f"✅ Loading model from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["g2_state_dict"])
model.eval()

# Load VGG for perceptual/style loss
vgg = VGG16FeatureExtractor().to(device).eval()

# Dataloader
test_loader = get_dataloader_g2(split="test")

# Storage
metrics_list = []

# Evaluation loop
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        input_img = batch["input_img"].to(device)
        guidance_img = batch["guidance_img"].to(device)
        mask = batch["mask"].to(device)
        gt_img = batch["gt_img"].to(device)

        pred_img = model(input_img, guidance_img, mask)
        pred_norm = (pred_img + 1) / 2
        gt_norm = gt_img

        pred_np = pred_norm.cpu().numpy().transpose(0, 2, 3, 1)
        gt_np = gt_norm.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(pred_np.shape[0]):
            psnr = compare_psnr(gt_np[i], pred_np[i], data_range=1.0)
            ssim = compare_ssim(gt_np[i], pred_np[i], channel_axis=-1, data_range=1.0)
            l1 = l1_loss(pred_img[i:i+1], gt_img[i:i+1]).item()
            perc = perceptual_loss(vgg, pred_img[i:i+1], gt_img[i:i+1]).item()
            style = style_loss(vgg, pred_img[i:i+1], gt_img[i:i+1]).item()

            metrics_list.append({
                "index": batch_idx * config.BATCH_SIZE + i,
                "psnr": psnr,
                "ssim": ssim,
                "l1_loss": l1,
                "perceptual_loss": perc,
                "style_loss": style
            })

            if save_images and i < 5:
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

                plt.tight_layout()
                fname = os.path.join(config.EVAL_RESULTS_DIR, f"sample_{batch_idx}_{i}.png")
                plt.savefig(fname)
                plt.close()

# Save metrics
df = pd.DataFrame(metrics_list)
csv_path = os.path.join(config.EVAL_RESULTS_DIR, "g2_evaluation_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"📊 Metrics saved to {csv_path}")

# Print mean/median
print("\n📈 Evaluation Summary:")
print(df.describe().loc[["mean", "50%"]].rename(index={"50%": "median"}))
