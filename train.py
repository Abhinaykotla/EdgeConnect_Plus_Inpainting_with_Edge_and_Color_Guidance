import time
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader import get_dataloader_g1
from g1_model import adversarial_loss, l1_loss
from g1_model import EdgeGenerator, EdgeDiscriminator
from config import config

# Function to save generated images (for training & validation)
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
    
    # Early Stopping Parameters
    patience = config.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    start_time = time.time()

    print(f"🔹 Starting Training for {num_epochs} Epochs on {config.DEVICE}...\n")
    for epoch in range(num_epochs):
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

            # Print progress every 50 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  🔹 Batch [{batch_idx+1}/{len(train_dataloader)}] - G1 Loss: {loss_g.item():.4f}, D1 Loss: {loss_d.item():.4f}")

        # **Save Training Samples Every 5 Epochs**
        if (epoch + 1) % 2 == 0:
            print(f"\n📸 Saving Training Samples for Epoch {epoch+1}...\n")
            save_generated_images(epoch+1, input_edges, mask, gt_edges, pred_edge, mode="train")

        ###### 🔹 Validation Phase ######
        if (epoch + 1) % 5 == 0:
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

        # End of Epoch: Print Loss and Time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\n🔹 Epoch [{epoch+1}/{num_epochs}] Completed in {epoch_duration:.2f}s - G1 Loss: {total_g_loss:.4f}, D1 Loss: {total_d_loss:.4f}\n")

    print(f"\n✅ Training Completed in {time.time() - start_time:.2f} seconds.\n")
