import os
import torch
import json
import glob
import hashlib
import io
import numpy as np
import matplotlib.pyplot as plt
from config import config

# Directory for saving G2 checkpoints
CHECKPOINT_DIR = config.MODEL_CHECKPOINT_DIR_G2
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_losses_to_json_g2(batch_losses, epoch_losses, save_dir):
    """
    Saves training loss data to JSON files with intelligent downsampling for G2 model.
    
    This function saves both batch-level and epoch-level loss histories to separate JSON files.
    It handles downsampling of large batch histories to prevent excessive file sizes.
    
    Args:
        batch_losses (dict): Dictionary containing batch-wise losses with keys like 'batch_idx', 'G2_L1', etc.
        epoch_losses (dict): Dictionary containing epoch-wise losses with keys like 'epoch', 'G2_Loss', 'D2_Loss'
        save_dir (str): Directory path to save the JSON files
        
    Returns:
        None: Data is saved to disk as JSON files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save batch losses with potential downsampling
    batch_path = os.path.join(save_dir, 'batch_losses.json')
    
    # Load existing data if available
    if os.path.exists(batch_path):
        try:
            with open(batch_path, 'r') as f:
                existing_batch_data = json.load(f)
                
                # Check if we need to downsample (if getting too large)
                total_points = len(existing_batch_data['batch_idx']) + len(batch_losses['batch_idx'])
                max_points = config.MAX_BATCH_POINTS  # Maximum points to store
                
                if total_points > max_points:
                    # Downsample the existing data to half
                    downsample_factor = 2
                    for key in existing_batch_data:
                        existing_batch_data[key] = existing_batch_data[key][::downsample_factor]
                    print(f"INFO: Downsampled batch history from {len(existing_batch_data['batch_idx'])*2} to {len(existing_batch_data['batch_idx'])} points")
                
                # Update with new data
                if existing_batch_data['batch_idx']:
                    # Calculate offset for new batch indices
                    last_batch_idx = existing_batch_data['batch_idx'][-1]
                    # Add offset to batch_idx
                    if batch_losses['batch_idx']:  # Check if not empty
                        next_idx = last_batch_idx + 1
                        updated_batch_idx = [next_idx + i for i, _ in enumerate(batch_losses['batch_idx'])]
                        batch_losses['batch_idx'] = updated_batch_idx
                
                # Append data
                for key in existing_batch_data:
                    if key in batch_losses and batch_losses[key]:  # Add only if not empty
                        existing_batch_data[key].extend(batch_losses[key])
                
                batch_losses = existing_batch_data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Could not load existing batch data: {e}")
    
    # Save updated batch data
    with open(batch_path, 'w') as f:
        json.dump(batch_losses, f)
    
    # Save epoch losses (leave as is)
    epoch_path = os.path.join(save_dir, 'epoch_losses.json')
    # Load existing data if available
    if os.path.exists(epoch_path):
        with open(epoch_path, 'r') as f:
            try:
                existing_epoch_data = json.load(f)
                # Merge epoch data without duplicates
                epoch_set = set(existing_epoch_data['epoch'])
                
                # Only add epochs that aren't already recorded
                for i, epoch in enumerate(epoch_losses['epoch']):
                    if epoch not in epoch_set:
                        existing_epoch_data['epoch'].append(epoch)
                        existing_epoch_data['G2_Loss'].append(epoch_losses['G2_Loss'][i])
                        existing_epoch_data['D2_Loss'].append(epoch_losses['D2_Loss'][i])
                
                epoch_losses = existing_epoch_data
            except (json.JSONDecodeError, KeyError):
                print("WARNING: Could not load existing epoch data, creating new file.")
    
    # Save updated epoch data
    with open(epoch_path, 'w') as f:
        json.dump(epoch_losses, f)


def plot_losses_g2(save_dir):
    """
    Plots training loss graphs from saved JSON files for G2 model.
    
    Creates a two-row figure:
    - Top: Batch-level losses for component-wise analysis
    - Bottom: Epoch-level losses for overall training progress
    
    Handles large loss files with file size checking to prevent memory issues.
    
    Args:
        save_dir (str): Directory containing the JSON loss files and where plots will be saved
        
    Returns:
        None: Plots are saved to disk as PNG files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load complete history from saved files for visualization
    batch_path = os.path.join(save_dir, 'batch_losses.json')
    epoch_path = os.path.join(save_dir, 'epoch_losses.json')
    
    batch_losses = {'batch_idx': [], 'G2_L1': [], 'G2_Adv': [], 'G2_FM': [], 'G2_Perc': [], 'G2_Style': [], 'D2_Real': [], 'D2_Fake': []}
    epoch_losses = {'epoch': [], 'G2_Loss': [], 'D2_Loss': []}
    
    if os.path.exists(batch_path):
        try:
            with open(batch_path, 'r') as f:
                batch_losses = json.load(f)
        except (json.JSONDecodeError, KeyError):
            print("WARNING: Could not load batch history.")
    
    if os.path.exists(epoch_path):
        try:
            with open(epoch_path, 'r') as f:
                epoch_losses = json.load(f)
        except (json.JSONDecodeError, KeyError):
            print("WARNING: Could not load epoch history, using empty data.")
    
    # Create plots
    plt.figure(figsize=(20, 12))
    
    # Plot 1: All batch losses (full top row)
    plt.subplot2grid((2, 1), (0, 0))
    
    # Only plot batch losses if the file isn't too large (for performance)
    if os.path.exists(batch_path) and os.path.getsize(batch_path) < 10_000_000:  # ~10MB limit
        try:
            if batch_losses['batch_idx']:
                plt.plot(batch_losses['batch_idx'], batch_losses['G2_L1'], label='G2 L1', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['G2_Adv'], label='G2 Adv', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['G2_FM'], label='G2 FM', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], 
                        [x / 10 for x in batch_losses['G2_Perc']], 
                        label='G2 Perc (scaled/10)', alpha=0.7)

                plt.plot(batch_losses['batch_idx'], batch_losses['G2_Style'], label='G2 Style', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['D2_Real'], label='D2 Real', linestyle='dashed', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['D2_Fake'], label='D2 Fake', linestyle='dashed', alpha=0.7)
        except (json.JSONDecodeError, KeyError, KeyError) as e:
            print(f"WARNING: Could not plot batch history: {e}")
    else:
        plt.text(0.5, 0.5, "Batch loss history too large to display", 
                horizontalalignment='center', verticalalignment='center')
            
    plt.xlabel('Global Batch Number')
    plt.ylabel('Loss Value')
    plt.title('G2 & D2 Batch Losses (Complete History)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: All epoch losses (full bottom row)
    plt.subplot2grid((2, 1), (1, 0))
    if epoch_losses['epoch']:
        plt.plot(epoch_losses['epoch'], epoch_losses['G2_Loss'], marker='o', label='G2 Loss', linewidth=2)
        plt.plot(epoch_losses['epoch'], epoch_losses['D2_Loss'], marker='s', label='D2 Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('G2 & D2 Epoch Losses (Complete History)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    latest_epoch = epoch_losses["epoch"][-1] if epoch_losses["epoch"] else "0"
    plt.savefig(os.path.join(save_dir, f'loss_trends_epoch_{latest_epoch}.png'))
    plt.close()


def save_generated_images_g2(epoch, masked_input, guidance, masks, gt_images, pred_images, save_dir=None, mode="train", batch_idx=None):
    """
    Saves visualization grids of G2 inpainting results for monitoring training progress.
    
    Creates a 2x3 visualization grid for each sample:
    - Row 1: Masked Input, Guidance Image, Generated Output
    - Row 2: Input Mask, Ground Truth, Visualization of Differences
    
    Args:
        epoch (int): Current epoch number
        masked_input (torch.Tensor): Input images with masks applied, shape [B, 3, H, W]
        guidance (torch.Tensor): Guidance images (edge and color maps), shape [B, 3, H, W]
        masks (torch.Tensor): Binary mask images, shape [B, 1, H, W]
        gt_images (torch.Tensor): Ground truth target images, shape [B, 3, H, W]
        pred_images (torch.Tensor): Generated inpainted images, shape [B, 3, H, W]
        save_dir (str, optional): Directory to save images (if None, uses config default)
        mode (str): Training mode ("train" or "val")
        batch_idx (int, optional): Batch index for batch-specific saves
        
    Returns:
        None: Images are saved to disk as PNG files
    """
    # Set up save directory
    if batch_idx is not None:
        batch_idx = batch_idx + 1  
        base_dir = os.path.join(config.BATCH_SAMPLES_DIR_G2, f"epoch_{epoch}")
        save_dir = os.path.join(base_dir, mode)
    else:
        save_dir = save_dir or os.path.join(config.EPOCH_SAMPLES_DIR_G2, mode)
    
    os.makedirs(save_dir, exist_ok=True)  
    batch_size = masked_input.shape[0]

    # Convert tensors to CPU for visualization
    masked_input = masked_input.cpu().detach()
    guidance = guidance.cpu().detach()
    masks = masks.cpu().detach()
    gt_images = gt_images.cpu().detach()
    pred_images = pred_images.cpu().detach()

    for i in range(min(batch_size, 5)):  # Save up to 5 samples
        fig = plt.figure(figsize=(15, 10))  # Create figure

        # Row 1
        plt.subplot(2, 3, 1)
        # Convert from tensor [C,H,W] to numpy [H,W,C] and normalize
        img = masked_input[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title("Masked Input")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        img = guidance[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title("Guidance Image")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        img = pred_images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title("Generated Output")
        plt.axis("off")

        # Row 2
        plt.subplot(2, 3, 4)
        plt.imshow(masks[i].squeeze(), cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        img = gt_images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title("Ground Truth")
        plt.axis("off")

        # Difference visualization
        plt.subplot(2, 3, 6)
        gt_np = gt_images[i].permute(1, 2, 0).numpy()
        pred_np = pred_images[i].permute(1, 2, 0).numpy()
        diff = np.abs(gt_np - pred_np)
        # Enhance differences for better visibility
        diff = diff / (diff.max() + 1e-8)
        plt.imshow(diff)
        plt.title("Differences")
        plt.axis("off")

        plt.tight_layout()

        # Save the figure
        filename = f"{mode}_epoch_{epoch}_batch_{batch_idx}_sample_{i}.png" if batch_idx else f"{mode}_epoch_{epoch}_sample_{i}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)


def save_checkpoint_g2(epoch, g2, d2, optimizer_g, optimizer_d, best_loss, history, batch_losses, epoch_losses, g2_ema=None):
    """
    Saves the G2/D2 model, optimizers, and training history as a checkpoint.
    
    Creates a comprehensive checkpoint file containing model weights, optimizer states,
    training history, and EMA parameters if available.
    
    Args:
        epoch (int): Current epoch number
        g2 (torch.nn.Module): Generator model (G2)
        d2 (torch.nn.Module): Discriminator model (D2)
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator
        best_loss (float): Best loss value achieved so far
        history (dict): Training history dictionary
        batch_losses (dict): Batch-wise loss history
        epoch_losses (dict): Epoch-wise loss history
        g2_ema (ExponentialMovingAverage, optional): EMA model for the generator
        
    Returns:
        None: Checkpoint is saved to disk
    """
    checkpoint = {
        "epoch": epoch,
        "g2_state_dict": g2.state_dict(),  # Save generator weights
        "d2_state_dict": d2.state_dict(),  # Save discriminator weights
        "optimizer_g": optimizer_g.state_dict(),  # Save generator optimizer state
        "optimizer_d": optimizer_d.state_dict(),  # Save discriminator optimizer state
        "best_loss": best_loss,  # Save the best loss value
        "history": history,  # Save training history
        "batch_losses": batch_losses,  # Save batch-wise losses
        "epoch_losses": epoch_losses,  # Save epoch-wise losses
    }

    # Save EMA shadow parameters if g2_ema is provided
    if g2_ema is not None:
        checkpoint["ema_shadow"] = g2_ema.shadow

    # Define the checkpoint path
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"INFO: Checkpoint saved: {checkpoint_path}")

    # Save training history separately for easier access
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"epochs": history, "batch_losses": batch_losses, "epoch_losses": epoch_losses}, f)

    # Manage old checkpoints (keep only the last 3)
    manage_checkpoints_g2()


def manage_checkpoints_g2():
    """
    Manages G2 model checkpoint files by removing older checkpoints.
    
    Retains only the most recent checkpoints (number defined in config.MAX_CHECKPOINTS)
    to prevent excessive disk usage while maintaining training continuity.
    
    Args:
        None: Uses global CHECKPOINT_DIR variable
        
    Returns:
        None: Modifies files on disk
    """
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    if len(checkpoint_files) > config.MAX_CHECKPOINTS:
        os.remove(checkpoint_files[0])  # Remove the oldest checkpoint
        print(f"INFO: Deleted old checkpoint: {checkpoint_files[0]}")


def load_checkpoint_g2(g2, d2, optimizer_g, optimizer_d, g2_ema=None):
    """
    Loads G2 model checkpoint with robust error handling and fallback mechanisms.
    
    Attempts to load the most recent checkpoint file. If unavailable, falls back to 
    training history JSON. Handles EMA weights if available.
    
    Args:
        g2 (torch.nn.Module): Generator model (G2) to load weights into
        d2 (torch.nn.Module): Discriminator model (D2) to load weights into
        optimizer_g (torch.optim.Optimizer): Generator optimizer to restore state
        optimizer_d (torch.optim.Optimizer): Discriminator optimizer to restore state
        g2_ema (ExponentialMovingAverage, optional): EMA model to restore shadow weights
        
    Returns:
        tuple: (start_epoch, best_loss, history, batch_losses, epoch_losses)
            - start_epoch (int): Next epoch to begin training from
            - best_loss (float): Best loss value achieved so far
            - history (dict): Training history dictionary
            - batch_losses (dict): Batch-wise loss history
            - epoch_losses (dict): Epoch-wise loss history
    """
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    batch_losses = {'batch_idx': [], 'G2_L1': [], 'G2_Adv': [], 'G2_FM': [], 'G2_Perc': [], 'G2_Style': [], 'D2_Real': [], 'D2_Fake': []}
    epoch_losses = {'epoch': [], 'G2_Loss': [], 'D2_Loss': []}

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint, map_location=config.DEVICE, weights_only=False)
        g2.load_state_dict(checkpoint["g2_state_dict"])
        d2.load_state_dict(checkpoint["d2_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]

        if "batch_losses" in checkpoint:
            batch_losses = checkpoint["batch_losses"]
        if "epoch_losses" in checkpoint:
            epoch_losses = checkpoint["epoch_losses"]

        if g2_ema is not None and "ema_shadow" in checkpoint:
            g2_ema.shadow = checkpoint["ema_shadow"]
            print("INFO: EMA weights restored.")

        start_epoch = checkpoint["epoch"] + 1
        print(f"INFO: Resuming training from epoch {start_epoch}, Best G2 Loss: {best_loss:.4f}")
        return start_epoch, best_loss, history, batch_losses, epoch_losses

    # Fallback to JSON if no checkpoint exists
    json_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                history = json_data.get("epochs", {"g2_loss": [], "d2_loss": []})
                if "batch_losses" in json_data:
                    batch_losses = json_data["batch_losses"]
                if "epoch_losses" in json_data:
                    epoch_losses = json_data["epoch_losses"]
                print("INFO: Loaded loss history from JSON file.")
        except (json.JSONDecodeError, KeyError):
            print("WARNING: Could not load history from JSON.")

    return 1, float("inf"), {"g2_loss": [], "d2_loss": []}, batch_losses, epoch_losses


def print_model_info_g2(model, model_name="Model"):
    """
    Prints detailed information about the G2 model architecture and parameters.
    
    Reports total parameter count, trainable parameters, and layer-by-layer breakdown
    to help understand model complexity and structure.
    
    Args:
        model (torch.nn.Module): PyTorch model to analyze
        model_name (str): Name to display for this model in the output
        
    Returns:
        tuple: (total_params, trainable_params)
            - total_params (int): Total number of parameters in the model
            - trainable_params (int): Number of trainable parameters in the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nINFO: {model_name} Summary")
    print(f"INFO: Total Parameters: {total_params:,}")
    print(f"INFO: Trainable Parameters: {trainable_params:,}")
    print("INFO: Layer-wise Breakdown:")

    for name, param in model.named_parameters():
        print(f"   {name}: {param.numel()} parameters")

    return total_params, trainable_params


def calculate_model_hash_g2(model):
    """
    Calculate a hash of model parameters to track changes.
    
    This function creates a unique hash based on model parameter shapes and values,
    providing a way to verify model state at different points during training.
    
    Args:
        model (torch.nn.Module): The PyTorch model to hash
        
    Returns:
        int: A hash value representing the current model state
    """
    params = [p.data for p in model.parameters()]
    param_shapes = [p.shape for p in params]
    param_data = torch.cat([p.flatten() for p in params if p.numel() > 0])
    param_hash = hash(str(param_shapes) + str(param_data.sum().item()) + str(param_data[:5].tolist()))
    return param_hash