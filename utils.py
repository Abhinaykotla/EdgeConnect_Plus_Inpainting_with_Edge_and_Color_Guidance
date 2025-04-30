import os
import torch
import json
import glob
import hashlib
import io
import matplotlib.pyplot as plt
from config import config

# Directory for saving checkpoints
CHECKPOINT_DIR = config.MODEL_CHECKPOINT_DIR_G1
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Function to save loss data to JSON files
def save_losses_to_json(batch_losses, epoch_losses, save_dir):
    """
    Saves batch and epoch losses with downsampling for large batch histories
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
                    print(f"Downsampled batch history from {len(existing_batch_data['batch_idx'])*2} to {len(existing_batch_data['batch_idx'])} points")
                
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
            print(f"Warning: Could not load existing batch data: {e}")
    
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
                        existing_epoch_data['G1_Loss'].append(epoch_losses['G1_Loss'][i])
                        existing_epoch_data['D1_Loss'].append(epoch_losses['D1_Loss'][i])
                
                epoch_losses = existing_epoch_data
            except (json.JSONDecodeError, KeyError):
                print("Warning: Could not load existing epoch data, creating new file.")
    
    # Save updated epoch data
    with open(epoch_path, 'w') as f:
        json.dump(epoch_losses, f)

# Modify the plot_losses function to only read from files and not save
def plot_losses(save_dir):
    """
    Plots loss graphs from JSON files with caching
    Layout: 2 plots
    - Top row: All batch losses (spanning full width)
    - Bottom row: All epoch losses (spanning full width)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load complete history from saved files for visualization
    batch_path = os.path.join(save_dir, 'batch_losses.json')
    epoch_path = os.path.join(save_dir, 'epoch_losses.json')
    
    batch_losses = {'batch_idx': [], 'G1_L1': [], 'G1_Adv': [], 'G1_FM': [], 'D1_Real': [], 'D1_Fake': []}
    epoch_losses = {'epoch': [], 'G1_Loss': [], 'D1_Loss': []}
    
    if os.path.exists(batch_path):
        try:
            with open(batch_path, 'r') as f:
                batch_losses = json.load(f)
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not load batch history.")
    
    if os.path.exists(epoch_path):
        try:
            with open(epoch_path, 'r') as f:
                epoch_losses = json.load(f)
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not load epoch history, using empty data.")
    
    # Create plots
    plt.figure(figsize=(20, 12))
    
    # Plot 1: All batch losses (full top row)
    plt.subplot2grid((2, 1), (0, 0))
    
    # Only plot batch losses if the file isn't too large (for performance)
    if os.path.exists(batch_path) and os.path.getsize(batch_path) < 10_000_000:  # ~10MB limit
        try:
            if batch_losses['batch_idx']:
                plt.plot(batch_losses['batch_idx'], batch_losses['G1_L1'], label='G1 L1', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['G1_Adv'], label='G1 Adv', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['G1_FM'], label='G1 FM', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['D1_Real'], label='D1 Real', linestyle='dashed', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['D1_Fake'], label='D1 Fake', linestyle='dashed', alpha=0.7)
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not load batch history.")
    else:
        plt.text(0.5, 0.5, "Batch loss history too large to display", 
                horizontalalignment='center', verticalalignment='center')
            
    plt.xlabel('Global Batch Number')
    plt.ylabel('Loss Value')
    plt.title('All Batch Losses (Complete History)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: All epoch losses (full bottom row)
    plt.subplot2grid((2, 1), (1, 0))
    if epoch_losses['epoch']:
        plt.plot(epoch_losses['epoch'], epoch_losses['G1_Loss'], marker='o', label='G1 Loss', linewidth=2)
        plt.plot(epoch_losses['epoch'], epoch_losses['D1_Loss'], marker='s', label='D1 Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Epoch Losses (Complete History)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    latest_epoch = epoch_losses["epoch"][-1] if epoch_losses["epoch"] else "0"
    plt.savefig(os.path.join(save_dir, f'loss_trends_epoch_{latest_epoch}.png'))
    # plt.savefig(os.path.join(save_dir, 'loss_trends_latest.png')) # Always overwrite this one for the latest view
    plt.close()

def save_generated_images(epoch, input_edges, masks, gt_edges, gray, pred_edges, save_dir=None, mode="train", batch_idx=None):
    """
    Saves generated images in a 1x2 grid:
    - Left (Big Box) → 2x2 subplots (Input Edges, Mask, GT Edges, Grayscale)
    - Right (Big Box) → 1 large subplot (Generated Edges)
    
    Args:
        epoch: Current epoch number
        input_edges: Input edge images
        gt_edges: Ground truth edge images
        pred_edges: Generated edge images
        masks: Mask images
        gray: Grayscale images
        save_dir: Base directory to save images
        mode: Training mode ("train" or "val")
        batch_idx: Optional batch index for batch-specific saves
    """
    # Set up save directory
    if batch_idx is not None:
        batch_idx = batch_idx + 1  
        base_dir = os.path.join(config.BATCH_SAMPLES_DIR_G1, f"epoch_{epoch}")
        save_dir = os.path.join(base_dir, mode)
    else:
        save_dir = save_dir or os.path.join(config.EPOCH_SAMPLES_DIR_G1, mode)
    
    os.makedirs(save_dir, exist_ok=True)  
    batch_size = input_edges.shape[0]

    # Convert tensors to CPU for visualization
    input_edges = input_edges.cpu().detach()
    masks = masks.cpu().detach()
    gt_edges = gt_edges.cpu().detach()
    pred_edges = pred_edges.cpu().detach()
    gray = gray.cpu().detach()

    for i in range(min(batch_size, 5)):  # Save up to 5 samples
        fig = plt.figure(figsize=(12, 6))  # Create figure

        # Define a 1x2 grid layout
        grid = fig.add_gridspec(1, 2, width_ratios=[1, 1])  # Left (1) | Right (1)

        # LEFT BIG BOX (Subdivided into 2x2)
        left_grid = grid[0].subgridspec(2, 2)  # Subdivide left into 2x2

        ax1 = fig.add_subplot(left_grid[0, 0])  # Top Left (Input Edges)
        ax1.imshow(input_edges[i].squeeze(), cmap="gray")
        ax1.set_title("Input Edges")
        ax1.axis("off")

        ax2 = fig.add_subplot(left_grid[0, 1])  # Top Right (Mask)
        ax2.imshow(masks[i].squeeze(), cmap="gray")
        ax2.set_title("Mask")
        ax2.axis("off")

        ax3 = fig.add_subplot(left_grid[1, 0])  # Bottom Left (GT Edges)
        ax3.imshow(gt_edges[i].squeeze(), cmap="gray")
        ax3.set_title("Ground Truth Edges")
        ax3.axis("off")

        ax4 = fig.add_subplot(left_grid[1, 1])  # Bottom Right (Grayscale)
        ax4.imshow(gray[i].squeeze(), cmap="gray")
        ax4.set_title("Grayscale Image")
        ax4.axis("off")

        # RIGHT BIG BOX (Generated Edges)
        ax5 = fig.add_subplot(grid[1])  # Right box (full height)
        ax5.imshow(pred_edges[i].squeeze(), cmap="gray")
        ax5.set_title(f"{mode.upper()} Generated")
        ax5.axis("off")

        plt.tight_layout()

        # Save the figure
        filename = f"{mode}_epoch_{epoch}_batch_{batch_idx}_sample_{i}.png" if batch_idx else f"{mode}_epoch_{epoch}_sample_{i}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)

# Function to save model and training history
def save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_loss, history, batch_losses, epoch_losses, g1_ema=None):
    """
    Saves the model, optimizers, and training history as a checkpoint.

    Args:
        epoch (int): Current epoch number.
        g1 (torch.nn.Module): Generator model.
        d1 (torch.nn.Module): Discriminator model.
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
        best_loss (float): Best loss value achieved so far.
        history (dict): Training history.
        batch_losses (dict): Batch-wise loss history.
        epoch_losses (dict): Epoch-wise loss history.
        g1_ema (ExponentialMovingAverage, optional): EMA model for the generator.
    """
    checkpoint = {
        "epoch": epoch,
        "g1_state_dict": g1.state_dict(),  # Save generator weights
        "d1_state_dict": d1.state_dict(),  # Save discriminator weights
        "optimizer_g": optimizer_g.state_dict(),  # Save generator optimizer state
        "optimizer_d": optimizer_d.state_dict(),  # Save discriminator optimizer state
        "best_loss": best_loss,  # Save the best loss value
        "history": history,  # Save training history
        "batch_losses": batch_losses,  # Save batch-wise losses
        "epoch_losses": epoch_losses,  # Save epoch-wise losses
    }

    # Save EMA shadow parameters if g1_ema is provided
    if g1_ema is not None:
        checkpoint["ema_shadow"] = g1_ema.shadow

    # Define the checkpoint path
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save training history separately for easier access
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"epochs": history, "batch_losses": batch_losses, "epoch_losses": epoch_losses}, f)

    # Manage old checkpoints (keep only the last 3)
    manage_checkpoints()

# Function to keep only the last 3 best checkpoints
def manage_checkpoints():
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    if len(checkpoint_files) > 3:
        os.remove(checkpoint_files[0])  # Remove the oldest checkpoint
        print(f"🗑️ Deleted old checkpoint: {checkpoint_files[0]}")

# Load checkpoint with enhanced data recovery
def load_checkpoint(g1, d1, optimizer_g, optimizer_d, g1_ema=None):
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    batch_losses = {'batch_idx': [], 'G1_L1': [], 'G1_Adv': [], 'G1_FM': [], 'D1_Real': [], 'D1_Fake': []}
    epoch_losses = {'epoch': [], 'G1_Loss': [], 'D1_Loss': []}

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint, map_location=config.DEVICE, weights_only=False)
        g1.load_state_dict(checkpoint["g1_state_dict"])
        d1.load_state_dict(checkpoint["d1_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]

        if "batch_losses" in checkpoint:
            batch_losses = checkpoint["batch_losses"]
        if "epoch_losses" in checkpoint:
            epoch_losses = checkpoint["epoch_losses"]

        if g1_ema is not None and "ema_shadow" in checkpoint:
            g1_ema.shadow = checkpoint["ema_shadow"]
            print("✅ EMA weights restored.")

        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resuming training from epoch {start_epoch}, Best G1 Loss: {best_loss:.4f}")
        return start_epoch, best_loss, history, batch_losses, epoch_losses

    # Fallback to JSON if no checkpoint exists
    json_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                history = json_data.get("epochs", {"g1_loss": [], "d1_loss": []})
                if "batch_losses" in json_data:
                    batch_losses = json_data["batch_losses"]
                if "epoch_losses" in json_data:
                    epoch_losses = json_data["epoch_losses"]
                print("🔄 Loaded loss history from JSON file.")
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not load history from JSON.")

    return 1, float("inf"), {"g1_loss": [], "d1_loss": []}, batch_losses, epoch_losses


def calculate_model_hash(model):
    """
    Calculate a hash for the model's state_dict to check for changes.
    """
    # Use BytesIO to save the model state to memory instead of None
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    # Get the bytes from the buffer
    buffer.seek(0)
    model_bytes = buffer.getvalue()
    # Calculate and return the hash
    return hashlib.md5(model_bytes).hexdigest()

def print_model_info(model, model_name="Model"):
    """
    Prints detailed information about the model, including:
    - Total number of parameters
    - Trainable parameters
    - Parameters per layer
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🔹 {model_name} Summary")
    print(f"🔹 Total Parameters: {total_params:,}")
    print(f"🔹 Trainable Parameters: {trainable_params:,}")
    print("🔹 Layer-wise Breakdown:")

    for name, param in model.named_parameters():
        print(f"   {name}: {param.numel()} parameters")

    return total_params, trainable_params