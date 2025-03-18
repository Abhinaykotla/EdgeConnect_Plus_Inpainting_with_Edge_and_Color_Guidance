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

    # Load checkpoint if available
    start_epoch, best_g1_loss, history = load_checkpoint(g1, d1, optimizer_g, optimizer_d)

    # Early Stopping Parameters
    patience = config.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    # Training Loop
    num_epochs = config.EPOCHS
    start_time = time.time()

    print(f"🔹 Starting Training for {num_epochs} Epochs on {config.DEVICE}...\n")
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
                g1_loss_l1 = l1_loss(pred_edge, gt_edges) * config.L1_LOSS_WEIGHT

                # **Adversarial Loss**
                fake_pred = d1(input_edges, pred_edge)  
                target_real = torch.ones_like(fake_pred, device=fake_pred.device)
                g1_loss_adv = adversarial_loss(fake_pred, target_real) * config.ADV_LOSS_WEIGHT

                loss_g = g1_loss_l1 + g1_loss_adv  

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            ###### 🔹 Train Discriminator (D1) ######
            optimizer_d.zero_grad()

            with torch.amp.autocast(config.DEVICE):  
                real_pred = d1(input_edges, gt_edges)  
                fake_pred_detached = d1(input_edges, pred_edge.detach())  

                target_fake = torch.zeros_like(fake_pred_detached, device=fake_pred_detached.device)

                real_loss = adversarial_loss(real_pred, target_real)
                fake_loss = adversarial_loss(fake_pred_detached, target_fake)
                loss_d = (real_loss + fake_loss) / 2  

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Track Losses
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()

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

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\n🔹 Epoch [{epoch}/{num_epochs}] Completed in {epoch_duration:.2f}s - G1 Loss: {avg_g1_loss:.4f}, D1 Loss: {avg_d1_loss:.4f}\n")

    print(f"\n✅ Training Completed in {time.time() - start_time:.2f} seconds.\n")
