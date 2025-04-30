import gc
from config import config
import torch
from train_loops import l1_loss
from g1_model import EdgeGenerator
from dataloader_g1 import get_dataloader_g1

def lr_finder(optimizer, model, dataloader, init_lr=1e-6, final_lr=1e-1, beta=0.98):
    """
    Learning Rate Finder for PyTorch.
    Helps determine the best learning rate by increasing LR exponentially.
    """
    
    # 🔹 1️⃣ Ensure Model is in Training Mode & Requires Gradients
    model.train()
    for param in model.parameters():
        param.requires_grad = True  # Ensure model parameters have gradients enabled

    # 🔹 2️⃣ Setup Learning Rate Scaling
    num = len(dataloader) - 1
    lr_multiplier = (final_lr / init_lr) ** (1 / num)
    optimizer.param_groups[0]['lr'] = init_lr

    avg_loss = 0.0
    best_loss = float('inf')
    losses = []
    log_lrs = []
    
    for batch_num, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        input_edges, gt_edges, mask = (
            batch["input_edge"].to(config.DEVICE),
            batch["gt_edge"].to(config.DEVICE),
            batch["mask"].to(config.DEVICE)
        )
        
        # 🔹 3️⃣ Compute Forward Pass
        pred_edge = model(input_edges, mask)
        loss = l1_loss(pred_edge, gt_edges)  # Use L1 loss for stability
        
        # 🔹 4️⃣ Track Best Loss Without Modifying Gradients
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(batch_num+1))

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        if smoothed_loss > 4 * best_loss:
            break  # Stop early if loss explodes

        losses.append(smoothed_loss)
        log_lrs.append(torch.log10(torch.tensor(optimizer.param_groups[0]['lr'])))

        # 🔹 5️⃣ Backward Pass & Optimizer Step
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        optimizer.param_groups[0]['lr'] *= lr_multiplier  # Increase LR exponentially

        # 🔹 6️⃣ Free GPU Memory After Each Batch
        del input_edges, gt_edges, mask, pred_edge, loss
        torch.cuda.empty_cache()
        gc.collect()

    return log_lrs, losses

import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("\n🔹 Finding the Best Learning Rate...\n")

    # Initialize model & optimizer
    g1 = EdgeGenerator().to(config.DEVICE)
    optimizer_g = torch.optim.Adam(g1.parameters(), lr=1e-6)  # Start with very low LR

    # Load data with smaller batch size (e.g., 64)
    train_dataloader_lr = get_dataloader_g1(split="train", use_mask=True)

    # Run the Learning Rate Finder
    log_lrs, losses = lr_finder(optimizer_g, g1, train_dataloader_lr)

    # Plot the Learning Rate Finder Results
    plt.plot(log_lrs, losses)
    plt.xlabel("Log Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")

    plt.savefig("lr_finder_plot.png")

    # Set Learning Rate Based on Best Loss
    config.LEARNING_RATE_G1 = 0.1 * 10 ** log_lrs[losses.index(min(losses))]
    print(f"✅ Best Learning Rate Selected: {config.LEARNING_RATE_G1:.6f}")