# train_loops.py

import time
import torch
from dataloader import get_dataloader_g1
from g1_model import adversarial_loss, l1_loss, feature_matching_loss, EdgeGenerator, EdgeDiscriminator
from config import config_g1
from utils import save_checkpoint, load_checkpoint, save_losses_to_json, plot_losses, save_generated_images

class EMA:
    """
    Exponential Moving Average for model weights.
    This helps produce more stable results by maintaining a moving average of model parameters.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters after each optimization step"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to the model for inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters to the model after inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Implements Gradient Penalty for WGAN-GP and helps stabilize training.
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=config_g1.DEVICE_G1)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, real_samples)

    grad_outputs = torch.ones(d_interpolates.size(), device=config_g1.DEVICE_G1)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Training Loop for EdgeConnect+ G1 Model
def train_g1_and_d1():
    """
    Main training loop for the EdgeConnect+ model.
    Trains the EdgeGenerator (G1) and EdgeDiscriminator (D1) models.
    """

    print("\n🔹 Initializing Model & Training Setup...\n")

    # Initialize models

    g1 = EdgeGenerator().to(config_g1.DEVICE_G1)
    d1 = EdgeDiscriminator().to(config_g1.DEVICE_G1)

    # Initialize EMA for G1 model with a decay rate of 0.999
    g1_ema = EMA(g1, decay=0.999)

    # Optimizers using config settings
    optimizer_g = torch.optim.Adam(
        g1.parameters(), 
        lr=config_g1.LEARNING_RATE_G1, 
        betas=(config_g1.BETA1_G1, config_g1.BETA2_G1), 
        weight_decay=config_g1.WEIGHT_DECAY_G1
    )

    optimizer_d = torch.optim.Adam(
        d1.parameters(), 
        lr=config_g1.LEARNING_RATE_G1 * config_g1.D2G_LR_RATIO_G1,  
        betas=(config_g1.BETA1_G1, config_g1.BETA2_G1), 
        weight_decay=config_g1.WEIGHT_DECAY_G1
    )

    # Use Mixed Precision for Faster Training
    scaler = torch.amp.GradScaler(device=config_g1.DEVICE_G1)

    print("Loading data into Dataloaders")
    # Load datasets
    train_dataloader = get_dataloader_g1(split="train", use_mask=True)
    val_dataloader = get_dataloader_g1(split="val", use_mask=True)  

    # Training Loop
    num_epochs = config_g1.EPOCHS_G1
    print(f"🔹 Training for a max of {num_epochs} Epochs on {config_g1.DEVICE_G1} with early stopping patience of {config_g1.EARLY_STOP_PATIENCE_G1} ...\n")

    # Load checkpoint if available
    start_epoch, best_g1_loss, history, batch_losses, epoch_losses = load_checkpoint(g1, d1, optimizer_g, optimizer_d)

    # Early Stopping Parameters
    patience = config_g1.EARLY_STOP_PATIENCE_G1
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        total_g_loss = 0.0
        total_d_loss = 0.0

        ###### 🔹 Training Phase ######
        for batch_idx, batch in enumerate(train_dataloader):
            input_edges, gt_edges, mask = (
                batch["input_edge"].to(config_g1.DEVICE_G1),   
                batch["gt_edge"].to(config_g1.DEVICE_G1),  
                batch["mask"].to(config_g1.DEVICE_G1)
            )

            ###### 🔹 Train Generator (G1) ######
            g1.train()
            optimizer_g.zero_grad()
            with torch.amp.autocast(config_g1.DEVICE_G1):  
                pred_edge = g1(input_edges, mask)  

                # L1 Loss
                g1_loss_l1 = l1_loss(pred_edge, gt_edges) * config_g1.L1_LOSS_WEIGHT_G1

                # Store these values for discriminator step
                pred_edge_detached = pred_edge.detach()
                
                # Adversarial Loss
                fake_pred = d1(input_edges, pred_edge)  
                target_real = torch.ones_like(fake_pred, device=config_g1.DEVICE_G1) * 0.9  # Smoothed labels
                g1_loss_adv = adversarial_loss(fake_pred, target_real)  

                # Feature Matching Loss
                real_features = d1(input_edges, gt_edges).detach()  # Real edge features from D1
                g1_loss_fm = feature_matching_loss(real_features, fake_pred) * config_g1.FM_LOSS_WEIGHT_G1  

                # Total Generator Loss
                loss_g = g1_loss_l1 + g1_loss_adv + g1_loss_fm

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()
            
            # Update EMA model after each generator update
            g1_ema.update()

            ###### 🔹 Train Discriminator (D1) ######
            optimizer_d.zero_grad()

            with torch.amp.autocast(config_g1.DEVICE_G1):  
                real_pred = d1(input_edges, gt_edges)  
                fake_pred_detached = d1(input_edges, pred_edge_detached)  # Use the detached tensor

                target_fake = torch.zeros_like(fake_pred_detached, device=config_g1.DEVICE_G1) + 0.1
                real_loss = adversarial_loss(real_pred, target_real)
                fake_loss = adversarial_loss(fake_pred_detached, target_fake)

                lambda_gp = 10  # Gradient penalty weight
                gp = gradient_penalty(d1, gt_edges, pred_edge_detached)
                loss_d = (real_loss + fake_loss) / 2 + lambda_gp * gp


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
            if (batch_idx + 1) % config_g1.BATCH_SAMPLING_SIZE_G1 == 0:
                print(f"  🔹 Batch [{batch_idx+1}/{len(train_dataloader)}] - G1 Loss: {loss_g.item():.4f}, D1 Loss: {loss_d.item():.4f}")

                print(f"\n📸 Saving Training Samples for batch {batch_idx+1}...\n")
                # Apply EMA for sample generation
                g1_ema.apply_shadow()
                with torch.no_grad():  # Add torch.no_grad() here for consistency
                    pred_edge_ema = g1(input_edges, mask)
                # Use epoch, not epoch+1 for consistent numbering
                save_generated_images(epoch, input_edges, mask, gt_edges, pred_edge_ema, mode="train", batch_idx=batch_idx+1)
                g1_ema.restore()

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
        save_losses_to_json(batch_losses, epoch_losses, config_g1.LOSS_PLOT_DIR_G1)
        
        # Reset batch losses for the next epoch to avoid duplication
        batch_losses = {'batch_idx': [], 'G1_L1': [], 'G1_Adv': [], 'G1_FM': [], 'D1_Real': [], 'D1_Fake': []}

        # Then plot using the saved JSON files
        plot_losses(config_g1.LOSS_PLOT_DIR_G1)

        # Save best model checkpoint if G1 loss improves
        if avg_g1_loss < best_g1_loss:
            best_g1_loss = avg_g1_loss
            # Apply EMA weights for saving the best model
            g1_ema.apply_shadow()
            # Don't pass the empty batch_losses here
            save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_g1_loss, history, batch_losses, epoch_losses)
            g1_ema.restore()
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            epochs_no_improve += 1
            
        # **Save Training Samples Every Epoch**
        if (epoch) % config_g1.TRAINING_SAMPLE_EPOCHS_G1 == 0:
            print(f"\n📸 Saving Training Samples for Epoch {epoch}...\n")
            # Apply EMA weights for visualization
            g1_ema.apply_shadow()
            with torch.no_grad():
                pred_edge_ema = g1(input_edges, mask)
                save_generated_images(
                epoch=epoch, 
                input_edges=input_edges, 
                gt_edges=gt_edges, 
                pred_edges=pred_edge_ema, 
                masks=mask,
                mode="train"
            )
            g1_ema.restore()

        ###### 🔹 Validation Phase ######
        if (epoch) % config_g1.VALIDATION_SAMPLE_EPOCHS_G1 == 0:
            print(f"\n🔍 Running Validation for Epoch {epoch}...\n")
            g1.eval()
            # Apply EMA weights for validation
            g1_ema.apply_shadow()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_input_edges, val_gt_edges, val_mask = (
                        val_batch["input_edge"].to(config_g1.DEVICE_G1),   
                        val_batch["gt_edge"].to(config_g1.DEVICE_G1),  
                        val_batch["mask"].to(config_g1.DEVICE_G1)
                    )

                    val_pred_edge = g1(val_input_edges, val_mask)

                    # Save validation images
                    save_generated_images(
                        epoch=epoch, 
                        input_edges=val_input_edges, 
                        gt_edges=val_gt_edges, 
                        pred_edges=val_pred_edge, 
                        masks=val_mask,
                        mode="val"
                    )
                    break  # Save only 1 batch per epoch
            g1_ema.restore()

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\n🔹 Epoch [{epoch}/{num_epochs}] Completed in {epoch_duration:.2f}s - G1 Loss: {avg_g1_loss:.4f}, D1 Loss: {avg_d1_loss:.4f}\n")

    print(f"\n✅ Training Completed in {time.time() - start_time:.2f} seconds.\n")

# import gc

# def lr_finder(optimizer, model, dataloader, init_lr=1e-6, final_lr=1e-1, beta=0.98):
#     """
#     Learning Rate Finder for PyTorch.
#     Helps determine the best learning rate by increasing LR exponentially.
#     """
    
#     # 🔹 1️⃣ Ensure Model is in Training Mode & Requires Gradients
#     model.train()
#     for param in model.parameters():
#         param.requires_grad = True  # Ensure model parameters have gradients enabled

#     # 🔹 2️⃣ Setup Learning Rate Scaling
#     num = len(dataloader) - 1
#     lr_multiplier = (final_lr / init_lr) ** (1 / num)
#     optimizer.param_groups[0]['lr'] = init_lr

#     avg_loss = 0.0
#     best_loss = float('inf')
#     losses = []
#     log_lrs = []
    
#     for batch_num, batch in enumerate(dataloader):
#         optimizer.zero_grad()
        
#         input_edges, gt_edges, mask = (
#             batch["input_edge"].to(config.DEVICE),
#             batch["gt_edge"].to(config.DEVICE),
#             batch["mask"].to(config.DEVICE)
#         )
        
#         # 🔹 3️⃣ Compute Forward Pass
#         pred_edge = model(input_edges, mask)
#         loss = l1_loss(pred_edge, gt_edges)  # Use L1 loss for stability
        
#         # 🔹 4️⃣ Track Best Loss Without Modifying Gradients
#         avg_loss = beta * avg_loss + (1 - beta) * loss.item()
#         smoothed_loss = avg_loss / (1 - beta**(batch_num+1))

#         if smoothed_loss < best_loss:
#             best_loss = smoothed_loss
#         if smoothed_loss > 4 * best_loss:
#             break  # Stop early if loss explodes

#         losses.append(smoothed_loss)
#         log_lrs.append(torch.log10(torch.tensor(optimizer.param_groups[0]['lr'])))

#         # 🔹 5️⃣ Backward Pass & Optimizer Step
#         loss.backward()  # Compute gradients
#         optimizer.step()  # Update parameters

#         optimizer.param_groups[0]['lr'] *= lr_multiplier  # Increase LR exponentially

#         # 🔹 6️⃣ Free GPU Memory After Each Batch
#         del input_edges, gt_edges, mask, pred_edge, loss
#         torch.cuda.empty_cache()
#         gc.collect()

#     return log_lrs, losses

# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     print("\n🔹 Finding the Best Learning Rate...\n")

#     # Initialize model & optimizer
#     g1 = EdgeGenerator().to(config.DEVICE)
#     optimizer_g = torch.optim.Adam(g1.parameters(), lr=1e-6)  # Start with very low LR

#     # Load data with smaller batch size (e.g., 64)
#     train_dataloader_lr = get_dataloader_g1(split="train", use_mask=True)

#     # Run the Learning Rate Finder
#     log_lrs, losses = lr_finder(optimizer_g, g1, train_dataloader_lr)

#     # Plot the Learning Rate Finder Results
#     plt.plot(log_lrs, losses)
#     plt.xlabel("Log Learning Rate")
#     plt.ylabel("Loss")
#     plt.title("Learning Rate Finder")

#     plt.savefig("lr_finder_plot.png")

#     # Set Learning Rate Based on Best Loss
#     config.LEARNING_RATE = 0.1 * 10 ** log_lrs[losses.index(min(losses))]
#     print(f"✅ Best Learning Rate Selected: {config.LEARNING_RATE:.6f}")
