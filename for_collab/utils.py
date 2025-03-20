# utils.py

import os
import torch
import json
import glob
import matplotlib.pyplot as plt
from config import config

# Directory for saving checkpoints
CHECKPOINT_DIR = config.MODEL_CHECKPOINT_DIR
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
                max_points = 5000  # Maximum points to store
                
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

def save_generated_images(epoch, input_edges, masks, gt_edges, pred_edges, save_dir=None, mode="train", batch_idx=None):
    """
    Saves generated images with an option for batch-specific directories.
    
    Args:
        epoch: Current epoch number
        input_edges: Input edge images
        gt_edges: Ground truth edge images
        pred_edges: Generated edge images
        masks: Mask images
        save_dir: Base directory to save images
        mode: Training mode ("train" or "val")
        batch_idx: Optional batch index for batch-specific saves
    """
    # Set up paths based on whether this is a batch save or epoch save
    if batch_idx is not None:
        # For batch saves, use a different directory structure
        batch_idx = batch_idx + 1  # Start indexing from 1
        base_dir = os.path.join(config.BATCH_SAMPLES_DIR, f"epoch_{epoch}")
        save_dir = os.path.join(base_dir, mode)
    else:
        # For epoch saves, use the traditional structure
        save_dir = save_dir or os.path.join(config.EPOCH_SAMPLES_DIR, mode)
    
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
    batch_size = input_edges.shape[0]

    # Normalize for visualization
    input_edges = input_edges.cpu().detach()
    masks = masks.cpu().detach()
    gt_edges = gt_edges.cpu().detach()
    pred_edges = pred_edges.cpu().detach()

    for i in range(min(batch_size, 5)):  # Save 5 sample images
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 2x2 grid of images

        # Input Edges - top left
        axes[0, 0].imshow(input_edges[i].squeeze(), cmap="gray")
        axes[0, 0].set_title(f"{mode.upper()} Input")
        axes[0, 0].axis("off")
        
        # Mask - top right
        axes[0, 1].imshow(masks[i].squeeze(), cmap="gray")
        axes[0, 1].set_title(f"{mode.upper()} Mask")
        axes[0, 1].axis("off")

        # Ground Truth Edges - bottom left
        axes[1, 0].imshow(gt_edges[i].squeeze(), cmap="gray")
        axes[1, 0].set_title(f"{mode.upper()} Ground Truth")
        axes[1, 0].axis("off")

        # Generated Edges - bottom right
        axes[1, 1].imshow(pred_edges[i].squeeze(), cmap="gray")
        axes[1, 1].set_title(f"{mode.upper()} Generated")
        axes[1, 1].axis("off")

        plt.tight_layout()

        # Save image with appropriate filename based on batch or epoch
        if batch_idx is not None:
            filename = f"{mode}_epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"
        else:
            filename = f"{mode}_epoch_{epoch}_sample_{i}.png"
            
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)  # Close figure to free memory

# Function to save model and training history
def save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_loss, history, batch_losses, epoch_losses):
    checkpoint = {
        "epoch": epoch,
        "g1_state_dict": g1.state_dict(),
        "d1_state_dict": d1.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "best_loss": best_loss,
        "history": history,
        "batch_losses": batch_losses,
        "epoch_losses": epoch_losses
    }

    # Save the checkpoint file
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save training history separately
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"epochs": history, "batch_losses": batch_losses, "epoch_losses": epoch_losses}, f)

    # Keep only the last 3 best checkpoints
    manage_checkpoints()

# Function to keep only the last 3 best checkpoints
def manage_checkpoints():
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    if len(checkpoint_files) > 3:
        os.remove(checkpoint_files[0])  # Remove the oldest checkpoint
        print(f"🗑️ Deleted old checkpoint: {checkpoint_files[0]}")

# Load checkpoint with enhanced data recovery
def load_checkpoint(g1, d1, optimizer_g, optimizer_d):
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    batch_losses = {'batch_idx': [], 'G1_L1': [], 'G1_Adv': [], 'G1_FM': [], 'D1_Real': [], 'D1_Fake': []}
    epoch_losses = {'epoch': [], 'G1_Loss': [], 'D1_Loss': []}
    
    if (checkpoint_files):
        latest_checkpoint = checkpoint_files[-1]  # Load most recent checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=config.DEVICE, weights_only=False)
        g1.load_state_dict(checkpoint["g1_state_dict"])
        d1.load_state_dict(checkpoint["d1_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]
        
        # Restore loss tracking if available
        if "batch_losses" in checkpoint:
            batch_losses = checkpoint["batch_losses"]
        if "epoch_losses" in checkpoint:
            epoch_losses = checkpoint["epoch_losses"]
            
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resuming training from epoch {start_epoch}, Best G1 Loss: {best_loss:.4f}")
        return start_epoch, best_loss, history, batch_losses, epoch_losses
    
    # Load from JSON if checkpoint isn't available but JSON history is
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