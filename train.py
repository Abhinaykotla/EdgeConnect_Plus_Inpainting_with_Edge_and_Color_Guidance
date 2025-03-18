import time
import os
import torch
import torch.nn.functional as F
import json
import glob
import matplotlib.pyplot as plt
from dataloader import get_dataloader_g1
from g1_model import adversarial_loss, l1_loss, EdgeGenerator, EdgeDiscriminator
from config import config

# Directory for saving checkpoints
CHECKPOINT_DIR = config.MODEL_CHECKPOINT_DIR
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_generated_images(epoch, input_edges, masks, gt_edges, pred_edges, save_dir="data_archive/generated_samples", mode="train"):
    os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
    batch_size = input_edges.shape[0]

    # Normalize for visualization
    input_edges = input_edges.cpu().detach()
    masks = masks.cpu().detach()
    gt_edges = gt_edges.cpu().detach()
    pred_edges = pred_edges.cpu().detach()

    for i in range(min(batch_size, 5)):  # Save 5 sample images
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))  # 4 images in a row

        # Input Edges
        axes[0].imshow(input_edges[i].squeeze(), cmap="gray")
        axes[0].set_title(f"{mode.upper()} Input")
        
        # Mask
        axes[1].imshow(masks[i].squeeze(), cmap="gray")
        axes[1].set_title(f"{mode.upper()} Mask")

        # Ground Truth Edges
        axes[2].imshow(gt_edges[i].squeeze(), cmap="gray")
        axes[2].set_title(f"{mode.upper()} Ground Truth")

        # Generated Edges
        axes[3].imshow(pred_edges[i].squeeze(), cmap="gray")
        axes[3].set_title(f"{mode.upper()} Generated")

        for ax in axes:
            ax.axis("off")

        # Save image
        save_path = os.path.join(save_dir, f"{mode}_epoch_{epoch}_sample_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)  # Close figure to free memory

# Function to save model and training history
def save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_loss, history):
    checkpoint = {
        "epoch": epoch,
        "g1_state_dict": g1.state_dict(),
        "d1_state_dict": d1.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "best_loss": best_loss,
        "history": history,
    }

    # Save the checkpoint file
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

    # Save training history separately
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    # Keep only the last 3 best checkpoints
    manage_checkpoints()

# Function to keep only the last 3 best checkpoints
def manage_checkpoints():
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    if len(checkpoint_files) > 3:
        os.remove(checkpoint_files[0])  # Remove the oldest checkpoint
        print(f"🗑️ Deleted old checkpoint: {checkpoint_files[0]}")

# Function to load checkpoint if available
def load_checkpoint(g1, d1, optimizer_g, optimizer_d):
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth")), key=os.path.getmtime)
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]  # Load most recent checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=config.DEVICE, weights_only=False)
        g1.load_state_dict(checkpoint["g1_state_dict"])
        d1.load_state_dict(checkpoint["d1_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔄 Resuming training from epoch {start_epoch}, Best G1 Loss: {best_loss:.4f}")
        return start_epoch, best_loss, history
    return 1, float("inf"), {"g1_loss": [], "d1_loss": []}

if __name__ == '__main__':
    print("\n🔹 Initializing Model & Training Setup...\n")

    # Initialize models
    g1 = EdgeGenerator().to(config.DEVICE)
    d1 = EdgeDiscriminator().to(config.DEVICE)

    # Optimizers using config settings
    optimizer_g = torch.optim.Adam(
        g1.parameters(), 
        lr=config.LEARNING_RATE, 
        betas=(config.BETA1, config.BETA2), 
        weight_decay=config.WEIGHT_DECAY
    )

    optimizer_d = torch.optim.Adam(
        d1.parameters(), 
        lr=config.LEARNING_RATE * config.D2G_LR_RATIO,  
        betas=(config.BETA1, config.BETA2), 
        weight_decay=config.WEIGHT_DECAY
    )

    # Use Mixed Precision for Faster Training
    scaler = torch.amp.GradScaler(device=config.DEVICE)

    # Load datasets
    train_dataloader = get_dataloader_g1(split="train", use_mask=True)
    val_dataloader = get_dataloader_g1(split="val", use_mask=True)  

    # Training Loop
    num_epochs = config.EPOCHS
    print(f"🔹 Training for {num_epochs} Epochs on {config.DEVICE}...\n")

    # Load checkpoint if available
    start_epoch, best_g1_loss, history = load_checkpoint(g1, d1, optimizer_g, optimizer_d)

    # Early Stopping Parameters
    patience = config.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        total_g_loss = 0.0
        total_d_loss = 0.0

        ###### 🔹 Training Phase ######
        for batch_idx, batch in enumerate(train_dataloader):
            input_edges, gt_edges, mask = (
                batch["input_edge"].to(config.DEVICE),   
                batch["gt_edge"].to(config.DEVICE),  
                batch["mask"].to(config.DEVICE)
            )

            ###### 🔹 Train Generator (G1) ######
            g1.train()
            optimizer_g.zero_grad()
            with torch.amp.autocast(config.DEVICE):  
                pred_edge = g1(input_edges, mask)  

                # **L1 Loss**
                g1_loss_l1 = l1_loss(pred_edge, gt_edges) * 100

                # **Adversarial Loss**
                fake_pred = d1(input_edges, pred_edge)  
                target_real = torch.ones_like(fake_pred, device=config.DEVICE) * 0.9  # Smoothed labels
                g1_loss_adv = adversarial_loss(fake_pred, target_real) * 1  

                loss_g = g1_loss_l1 + g1_loss_adv  

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            ###### 🔹 Train Discriminator (D1) ######
            optimizer_d.zero_grad()

            with torch.amp.autocast(config.DEVICE):  
                real_pred = d1(input_edges, gt_edges)  
                fake_pred_detached = d1(input_edges, pred_edge.detach())  

                target_fake = torch.zeros_like(fake_pred_detached, device=config.DEVICE) + 0.1
                real_loss = adversarial_loss(real_pred, target_real)
                fake_loss = adversarial_loss(fake_pred_detached, target_fake)

                loss_d = 0.5 * (real_loss + fake_loss)  # Weighted loss

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Track Losses
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  🔹 Batch [{batch_idx+1}/{len(train_dataloader)}] - G1 Loss: {loss_g.item():.4f}, D1 Loss: {loss_d.item():.4f}")

        # Compute average loss for the epoch
        avg_g1_loss = total_g_loss / len(train_dataloader)
        avg_d1_loss = total_d_loss / len(train_dataloader)

        # Save training history
        history["g1_loss"].append(avg_g1_loss)
        history["d1_loss"].append(avg_d1_loss)

        # Save best model checkpoint if G1 loss improves
        if avg_g1_loss < best_g1_loss:
            best_g1_loss = avg_g1_loss
            save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_g1_loss, history)
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            epochs_no_improve += 1
            
        # **Save Training Samples Every 5 Epochs**
        if (epoch + 1) % config.TRAINING_SAMPLE_EPOCHS == 0:
            print(f"\n📸 Saving Training Samples for Epoch {epoch+1}...\n")
            save_generated_images(epoch+1, input_edges, mask, gt_edges, pred_edge, mode="train")

        ###### 🔹 Validation Phase ######
        if (epoch + 1) % config.VALIDATION_SAMPLE_EPOCHS == 0:
            print(f"\n🔍 Running Validation for Epoch {epoch+1}...\n")
            g1.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_input_edges, val_gt_edges, val_mask = (
                        val_batch["input_edge"].to(config.DEVICE),   
                        val_batch["gt_edge"].to(config.DEVICE),  
                        val_batch["mask"].to(config.DEVICE)
                    )

                    val_pred_edge = g1(val_input_edges, val_mask)

                    # Save validation images
                    save_generated_images(epoch+1, val_input_edges, val_gt_edges, val_pred_edge, val_mask, mode="val")
                    break  # Save only 1 batch per epoch

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\n🔹 Epoch [{epoch}/{num_epochs}] Completed in {epoch_duration:.2f}s - G1 Loss: {avg_g1_loss:.4f}, D1 Loss: {avg_d1_loss:.4f}\n")

    print(f"\n✅ Training Completed in {time.time() - start_time:.2f} seconds.\n")
