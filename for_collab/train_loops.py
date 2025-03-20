# train_loops.py

import time
import torch
from dataloader import get_dataloader_g1
from g1_model import adversarial_loss, l1_loss, feature_matching_loss, EdgeGenerator, EdgeDiscriminator
from config import config
from utils import save_checkpoint, load_checkpoint, save_losses_to_json, plot_losses, save_generated_images

def train_g1_and_d1():
    """
    Main training loop for the EdgeConnect+ model.
    Trains the EdgeGenerator (G1) and EdgeDiscriminator (D1) models.
    """

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

    print("Loading data into Dataloaders")
    # Load datasets
    train_dataloader = get_dataloader_g1(split="train", use_mask=True)
    val_dataloader = get_dataloader_g1(split="val", use_mask=True)  

    # Training Loop
    num_epochs = config.EPOCHS
    print(f"🔹 Training for a max of {num_epochs} Epochs on {config.DEVICE} with early stopping patience of {config.EARLY_STOP_PATIENCE} ...\n")

    # Load checkpoint if available
    start_epoch, best_g1_loss, history, batch_losses, epoch_losses = load_checkpoint(g1, d1, optimizer_g, optimizer_d)

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

                # L1 Loss
                g1_loss_l1 = l1_loss(pred_edge, gt_edges) * config.L1_LOSS_WEIGHT  

                # Store these values for discriminator step
                pred_edge_detached = pred_edge.detach()
                
                # Adversarial Loss
                fake_pred = d1(input_edges, pred_edge)  
                target_real = torch.ones_like(fake_pred, device=config.DEVICE) * 0.9  # Smoothed labels
                g1_loss_adv = adversarial_loss(fake_pred, target_real)  

                # Feature Matching Loss
                real_features = d1(input_edges, gt_edges).detach()  # Real edge features from D1
                g1_loss_fm = feature_matching_loss(real_features, fake_pred) * config.FM_LOSS_WEIGHT  

                # Total Generator Loss
                loss_g = g1_loss_l1 + g1_loss_adv + g1_loss_fm

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            ###### 🔹 Train Discriminator (D1) ######
            optimizer_d.zero_grad()

            with torch.amp.autocast(config.DEVICE):  
                real_pred = d1(input_edges, gt_edges)  
                fake_pred_detached = d1(input_edges, pred_edge_detached)  # Use the detached tensor

                target_fake = torch.zeros_like(fake_pred_detached, device=config.DEVICE) + 0.1
                real_loss = adversarial_loss(real_pred, target_real)
                fake_loss = adversarial_loss(fake_pred_detached, target_fake)

                loss_d = (real_loss + fake_loss) / 2 

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Track Losses
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            batch_losses['batch_idx'].append(batch_idx)
            batch_losses['G1_L1'].append(g1_loss_l1.item())
            batch_losses['G1_Adv'].append(g1_loss_adv.item())
            batch_losses['G1_FM'].append(g1_loss_fm.item())
            batch_losses['D1_Real'].append(real_loss.item())
            batch_losses['D1_Fake'].append(fake_loss.item())

            # Print progress every 100 batches
            if (batch_idx + 1) % config.BATCH_SAMPLING_SIZE == 0:
                print(f"  🔹 Batch [{batch_idx+1}/{len(train_dataloader)}] - G1 Loss: {loss_g.item():.4f}, D1 Loss: {loss_d.item():.4f}")

                print(f"\n📸 Saving Training Samples for batch {batch_idx+1}...\n")
                save_generated_images(epoch, input_edges, mask, gt_edges, pred_edge, mode="train", batch_idx=batch_idx)

        # Compute average loss for the epoch
        avg_g1_loss = total_g_loss / len(train_dataloader)
        avg_d1_loss = total_d_loss / len(train_dataloader)
        epoch_losses['epoch'].append(epoch)
        epoch_losses['G1_Loss'].append(avg_g1_loss)
        epoch_losses['D1_Loss'].append(avg_d1_loss)

        # Save training history
        history["g1_loss"].append(avg_g1_loss)
        history["d1_loss"].append(avg_d1_loss)

        # First save the current losses to JSON files
        save_losses_to_json(batch_losses, epoch_losses, config.LOSS_PLOT_DIR)
        
        # Reset batch losses for the next epoch to avoid duplication
        batch_losses = {'batch_idx': [], 'G1_L1': [], 'G1_Adv': [], 'G1_FM': [], 'D1_Real': [], 'D1_Fake': []}

        # Then plot using the saved JSON files
        plot_losses(config.LOSS_PLOT_DIR)

        # Save best model checkpoint if G1 loss improves
        if avg_g1_loss < best_g1_loss:
            best_g1_loss = avg_g1_loss
            save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_g1_loss, history, batch_losses, epoch_losses)
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            epochs_no_improve += 1
            
        # **Save Training Samples Every Epoch**
        if (epoch) % config.TRAINING_SAMPLE_EPOCHS == 0:
            print(f"\n📸 Saving Training Samples for Epoch {epoch}...\n")
            save_generated_images(
                epoch=epoch+1, 
                input_edges=input_edges, 
                gt_edges=gt_edges, 
                pred_edges=pred_edge, 
                masks=mask,
                mode="train"
            )

        ###### 🔹 Validation Phase ######
        if (epoch) % config.VALIDATION_SAMPLE_EPOCHS == 0:
            print(f"\n🔍 Running Validation for Epoch {epoch}...\n")
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
                    save_generated_images(
                        epoch=epoch+1, 
                        input_edges=val_input_edges, 
                        gt_edges=val_gt_edges, 
                        pred_edges=val_pred_edge, 
                        masks=val_mask,
                        mode="val"
                    )
                    break  # Save only 1 batch per epoch

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\n🔹 Epoch [{epoch}/{num_epochs}] Completed in {epoch_duration:.2f}s - G1 Loss: {avg_g1_loss:.4f}, D1 Loss: {avg_d1_loss:.4f}\n")

    print(f"\n✅ Training Completed in {time.time() - start_time:.2f} seconds.\n")
