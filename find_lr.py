import gc
from config import config
import torch
from loss_functions import l1_loss
from g1_model import EdgeGenerator
from dataloader_g1 import get_dataloader_g1

def lr_finder(optimizer, model, dataloader, init_lr=1e-6, final_lr=1e-1, beta=0.98):
    """
    Learning Rate Finder for PyTorch that implements the cyclical learning rate technique.
    
    Uses exponential learning rate increase and tracks changes in loss values to find
    the optimal learning rate before loss destabilizes.
    
    Args:
        optimizer (torch.optim.Optimizer): Model optimizer to update learning rate
        model (torch.nn.Module): PyTorch model to train
        dataloader (torch.utils.data.DataLoader): DataLoader providing training data
        init_lr (float): Starting learning rate (very small value, default: 1e-6)
        final_lr (float): Maximum learning rate to try (default: 1e-1)
        beta (float): Exponential moving average smoothing factor for loss (default: 0.98)
    
    Returns:
        tuple: (log_lrs, losses) - Lists of log learning rates and corresponding smoothed losses
    """
    
    # Ensure Model is in Training Mode & Requires Gradients
    model.train()
    for param in model.parameters():
        param.requires_grad = True  # Ensure model parameters have gradients enabled

    # Setup Learning Rate Scaling
    num = len(dataloader) - 1
    lr_multiplier = (final_lr / init_lr) ** (1 / num)  # Factor to multiply LR each iteration
    optimizer.param_groups[0]['lr'] = init_lr

    # Initialize tracking variables
    avg_loss = 0.0  # Exponential moving average of loss
    best_loss = float('inf')  # Track best loss for early stopping
    losses = []  # Store smoothed losses for plotting
    log_lrs = []  # Store log learning rates for plotting
    
    for batch_num, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Extract data from batch and move to device
        # Shape: [B, 1, H, W] for each tensor (edge maps and binary mask)
        input_edges, gt_edges, mask = (
            batch["input_edge"].to(config.DEVICE),
            batch["gt_edge"].to(config.DEVICE),
            batch["mask"].to(config.DEVICE)
        )
        
        # Compute Forward Pass
        pred_edge = model(input_edges, mask)  # Shape: [B, 1, H, W]
        loss = l1_loss(pred_edge, gt_edges)  # Use L1 loss for stability
        
        # Track Best Loss With Exponential Moving Average
        # Smoothes the loss to reduce impact of noisy batches
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        # Correct bias in early iterations
        smoothed_loss = avg_loss / (1 - beta**(batch_num+1))

        # Early stopping condition: if loss is growing significantly
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        if smoothed_loss > 4 * best_loss:
            break  # Stop early if loss explodes (4x best loss threshold)

        losses.append(smoothed_loss)
        log_lrs.append(torch.log10(torch.tensor(optimizer.param_groups[0]['lr'])))

        # Backward Pass & Optimizer Step
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        # Increase LR exponentially for next iteration
        optimizer.param_groups[0]['lr'] *= lr_multiplier  

        # Free GPU Memory After Each Batch
        # This helps prevent OOM errors during the search
        del input_edges, gt_edges, mask, pred_edge, loss
        torch.cuda.empty_cache()
        gc.collect()

    return log_lrs, losses

import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("\nINFO: Finding the Best Learning Rate...\n")

    # Initialize model & optimizer
    g1 = EdgeGenerator().to(config.DEVICE)
    optimizer_g = torch.optim.Adam(g1.parameters(), lr=1e-6)  # Start with very low LR

    # Load data with smaller batch size (e.g., 64)
    # Using training data to find an appropriate learning rate
    train_dataloader_lr = get_dataloader_g1(split="train", use_mask=True)

    # Run the Learning Rate Finder
    log_lrs, losses = lr_finder(optimizer_g, g1, train_dataloader_lr)

    # Plot the Learning Rate Finder Results
    # The optimal learning rate is typically just before the loss begins to increase
    plt.plot(log_lrs, losses)
    plt.xlabel("Log Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")

    plt.savefig("lr_finder_plot.png")

    # Set Learning Rate Based on Best Loss
    # Using a fraction (0.1) of the learning rate at minimum loss
    # This is a common practice to ensure stability
    config.LEARNING_RATE_G1 = 0.1 * 10 ** log_lrs[losses.index(min(losses))]
    print(f"RESULT: Best Learning Rate Selected: {config.LEARNING_RATE_G1:.6f}")