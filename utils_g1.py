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

def save_losses_to_json(batch_losses, epoch_losses, save_dir):
    """
    Saves training loss data to JSON files with intelligent downsampling for G1 model.
    
    This function saves both batch-level and epoch-level loss histories to separate JSON files.
    It implements downsampling for large batch histories to prevent excessive file sizes,
    and manages the merging of new data with existing data.
    
    Args:
        batch_losses (dict): Dictionary containing batch-wise losses with keys like 'batch_idx', 'G1_L1', etc.
        epoch_losses (dict): Dictionary containing epoch-wise losses with keys like 'epoch', 'G1_Loss', 'D1_Loss'
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
                    # Downsample the existing data to half by taking every other point
                    downsample_factor = 2
                    for key in existing_batch_data:
                        existing_batch_data[key] = existing_batch_data[key][::downsample_factor]
                    print(f"INFO: Downsampled batch history from {len(existing_batch_data['batch_idx'])*2} to {len(existing_batch_data['batch_idx'])} points")
                
                # Update with new data
                if existing_batch_data['batch_idx']:
                    # Calculate offset for new batch indices to ensure continuous numbering
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
                        existing_epoch_data['G1_Loss'].append(epoch_losses['G1_Loss'][i])
                        existing_epoch_data['D1_Loss'].append(epoch_losses['D1_Loss'][i])
                
                epoch_losses = existing_epoch_data
            except (json.JSONDecodeError, KeyError):
                print("WARNING: Could not load existing epoch data, creating new file.")
    
    # Save updated epoch data
    with open(epoch_path, 'w') as f:
        json.dump(epoch_losses, f)


def plot_losses(save_dir):
    """
    Plots training loss graphs from saved JSON files for G1 model.
    
    Creates a two-row figure displaying:
    - Top row: Batch-wise losses for all components (L1, Adv, FM, VGG, etc.)
    - Bottom row: Epoch-level losses showing G1 and D1 average losses
    
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
    
    # Initialize empty dictionaries with expected keys
    batch_losses = {'batch_idx': [], 'G1_L1': [], 'G1_Adv': [], 'G1_FM': [], 'G1_VGG': [], 'D1_Real': [], 'D1_Fake': []}
    epoch_losses = {'epoch': [], 'G1_Loss': [], 'D1_Loss': []}
    
    # Try to load batch losses from file
    if os.path.exists(batch_path):
        try:
            with open(batch_path, 'r') as f:
                batch_losses = json.load(f)
        except (json.JSONDecodeError, KeyError):
            print("WARNING: Could not load batch history.")
    
    # Try to load epoch losses from file
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
                plt.plot(batch_losses['batch_idx'], batch_losses['G1_L1'], label='G1 L1', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['G1_Adv'], label='G1 Adv', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['G1_FM'], label='G1 FM', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['G1_VGG'], label='G1 VGG', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['D1_Real'], label='D1 Real', linestyle='dashed', alpha=0.7)
                plt.plot(batch_losses['batch_idx'], batch_losses['D1_Fake'], label='D1 Fake', linestyle='dashed', alpha=0.7)
        except (json.JSONDecodeError, KeyError):
            print("WARNING: Could not load batch history.")
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
    plt.close()


def save_generated_images(epoch, input_edges, masks, gt_edges, gray, pred_edges, save_dir=None, mode="train", batch_idx=None):
    """
    Saves visualization grids of G1 edge generation results for monitoring training progress.
    
    Creates a 1x2 grid visualization for each sample:
    - Left: 2x2 subplots (Input Edges, Mask, GT Edges, Grayscale)
    - Right: 1 large subplot (Generated Edges)
    
    Args:
        epoch (int): Current epoch number
        input_edges (torch.Tensor): Input edge images with holes, shape [B, 1, H, W]
        masks (torch.Tensor): Binary mask images, shape [B, 1, H, W]
        gt_edges (torch.Tensor): Ground truth edge images, shape [B, 1, H, W]
        gray (torch.Tensor): Grayscale images for guidance, shape [B, 1, H, W]
        pred_edges (torch.Tensor): Generated edge images, shape [B, 1, H, W]
        save_dir (str, optional): Directory to save images (if None, uses config default)
        mode (str): Training mode ("train" or "val")
        batch_idx (int, optional): Batch index for batch-specific saves
        
    Returns:
        None: Images are saved to disk as PNG files
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


def save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_loss, history, batch_losses, epoch_losses, g1_ema=None):
    """
    Saves the G1/D1 model, optimizers, and training history as a checkpoint.
    
    Creates a comprehensive checkpoint file containing model weights, optimizer states,
    training history, and EMA parameters if available.
    
    Args:
        epoch (int): Current epoch number
        g1 (torch.nn.Module): Generator model (G1)
        d1 (torch.nn.Module): Discriminator model (D1)
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator
        best_loss (float): Best loss value achieved so far
        history (dict): Training history dictionary
        batch_losses (dict): Batch-wise loss history
        epoch_losses (dict): Epoch-wise loss history
        g1_ema (ExponentialMovingAverage, optional): EMA model for the generator
        
    Returns:
        None: Checkpoint is saved to disk
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
    print(f"INFO: Checkpoint saved: {checkpoint_path}")

    # Save training history separately for easier access
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"epochs": history, "batch_losses": batch_losses, "epoch_losses": epoch_losses}, f)

    # Manage old checkpoints (keep only the last MAX_CHECKPOINTS)
    manage_checkpoints()


def manage_checkpoints():
    """
    Manages G1 model checkpoint files by removing older checkpoints.
    
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


def load_checkpoint(g1, d1, optimizer_g, optimizer_d, g1_ema=None):
    """
    Loads G1 model checkpoint with robust error handling and fallback mechanisms.
    
    Attempts to load the most recent checkpoint file. If unavailable, falls back to 
    training history JSON. Handles EMA weights if available. Can optionally override
    learning rates from config.
    
    Args:
        g1 (torch.nn.Module): Generator model (G1) to load weights into
        d1 (torch.nn.Module): Discriminator model (D1) to load weights into
        optimizer_g (torch.optim.Optimizer): Generator optimizer to restore state
        optimizer_d (torch.optim.Optimizer): Discriminator optimizer to restore state
        g1_ema (ExponentialMovingAverage, optional): EMA model to restore shadow weights
        
    Returns:
        tuple: (start_epoch, best_loss, history, batch_losses, epoch_losses)
            - start_epoch (int): Next epoch to begin training from
            - best_loss (float): Best loss value achieved so far
            - history (dict): Training history dictionary
            - batch_losses (dict): Batch-wise loss history
            - epoch_losses (dict): Epoch-wise loss history
    """
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

        # Force override learning rates from config
        if config.OVERRIDE_LR:
            for param_group in optimizer_g.param_groups:
                param_group["lr"] = config.LEARNING_RATE_G1
            for param_group in optimizer_d.param_groups:
                param_group["lr"] = config.LEARNING_RATE_G1 * config.D2G_LR_RATIO_G1

        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]

        if "batch_losses" in checkpoint:
            batch_losses = checkpoint["batch_losses"]
        if "epoch_losses" in checkpoint:
            epoch_losses = checkpoint["epoch_losses"]

        if g1_ema is not None and "ema_shadow" in checkpoint:
            g1_ema.shadow = checkpoint["ema_shadow"]
            print("INFO: EMA weights restored.")

        start_epoch = checkpoint["epoch"] + 1
        print(f"INFO: Resuming training from epoch {start_epoch}, Best G1 Loss: {best_loss:.4f}")
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
                print("INFO: Loaded loss history from JSON file.")
        except (json.JSONDecodeError, KeyError):
            print("WARNING: Could not load history from JSON.")

    return 1, float("inf"), {"g1_loss": [], "d1_loss": []}, batch_losses, epoch_losses


def calculate_model_hash(model):
    """
    Calculate a hash of model parameters to track changes.
    
    This function creates a unique hash based on model parameters by serializing 
    the model's state dictionary and computing an MD5 hash. This is useful for 
    verifying model state at different points during training.
    
    Args:
        model (torch.nn.Module): The PyTorch model to hash
        
    Returns:
        str: A hash string representing the current model state
    """
    # Use BytesIO to save the model state to memory instead of disk
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    # Get the bytes from the buffer
    buffer.seek(0)
    model_bytes = buffer.getvalue()
    # Calculate and return the hash
    return hashlib.md5(model_bytes).hexdigest()


def print_model_info(model, model_name="Model"):
    """
    Prints detailed information about the model architecture and parameters.
    
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