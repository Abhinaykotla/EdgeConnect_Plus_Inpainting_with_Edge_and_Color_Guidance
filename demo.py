
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
from dataloader_g2 import get_dataloader_g2
from g2_model import InpaintingGeneratorG2



def main():
    device = config.DEVICE
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    save_images = True

    # Load model
    model = InpaintingGeneratorG2().to(device)
    checkpoint_path = config.G2_MODEL_PATH
    print(f"✅ Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["g2_state_dict"])
    model.eval()

    # Load test data
    test_loader = get_dataloader_g2(split="demo", batch_size=config.BATCH_SIZE_G2_INFERENCE)
    metrics_list = []

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

                if save_images and i < 2:
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
    csv_path = os.path.join(config.EVAL_RESULTS_DIR, "evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"📊 Metrics saved to: {csv_path}")

    # Summary
    print("\n📈 Evaluation Summary:")
    print(df.describe().loc[["mean", "50%"]].rename(index={"50%": "median"}))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
