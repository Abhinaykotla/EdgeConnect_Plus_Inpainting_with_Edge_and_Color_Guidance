# train_loops.py

import time
import torch
from dataloader_g1 import get_dataloader_g1
from g1_model import adversarial_loss, l1_loss, feature_matching_loss, EdgeGenerator, EdgeDiscriminator
from utils import save_checkpoint, load_checkpoint, save_losses_to_json, plot_losses, save_generated_images, print_model_info, calculate_model_hash
from config import config

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
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=config.DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, real_samples)

    grad_outputs = torch.ones(d_interpolates.size(), device=config.DEVICE)
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
    g1 = EdgeGenerator().to(config.DEVICE)
    d1 = EdgeDiscriminator().to(config.DEVICE)
    
    # Initialize EMA for G1 model with a decay rate of 0.999
    g1_ema = EMA(g1, decay=0.999)

    # Optimizers using config settings
    optimizer_g = torch.optim.Adam(
        g1.parameters(), 
        lr=config.LEARNING_RATE_G1, 
        betas=(config.BETA1, config.BETA2), 
        weight_decay=config.WEIGHT_DECAY
    )

    optimizer_d = torch.optim.Adam(
        d1.parameters(), 
        lr=config.LEARNING_RATE_G1 * config.D2G_LR_RATIO_G1,  
        betas=(config.BETA1, config.BETA2), 
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.7)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.7)

    # Use Mixed Precision for Faster Training
    scaler = torch.amp.GradScaler(device=config.DEVICE)

    print("Loading data into Dataloaders")
    # Load datasets
    train_dataloader = get_dataloader_g1(split="train", use_mask=True)
    val_dataloader = get_dataloader_g1(split="val", use_mask=True)  

    # Training Loop
    num_epochs = config.EPOCHS
    print(f"🔹 Training for a max of {num_epochs} Epochs on {config.DEVICE} with early stopping patience of {config.EARLY_STOP_PATIENCE} ...\n")

    # Print loss weights
    print(f"🔹 Loss Weights → L1: {config.L1_LOSS_WEIGHT}, Adv: {config.ADV_LOSS_WEIGHT}, FM: {config.FM_LOSS_WEIGHT}")

    print("🔹 Checking for old checkpoints\n")
    print("Model Hash before loading:", calculate_model_hash(g1)) 

    # Load checkpoint if available
    start_epoch, best_g1_loss, history, batch_losses, epoch_losses = load_checkpoint(g1, d1, optimizer_g, optimizer_d, g1_ema)

    print("Model Hash after loading:", calculate_model_hash(g1)) 

    # Early Stopping Parameters
    patience = config.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    start_time = time.time()

    # Example: Print information for G1 (Generator) and D1 (Discriminator)
    print_model_info(g1, model_name="Generator (G1)")
    print_model_info(d1, model_name="Discriminator (D1)")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        total_g_loss = 0.0
        total_d_loss = 0.0

        # Print current learning rates
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        print(f"🔹 Current Learning Rates → G1: {current_lr_g:.9f}, D1: {current_lr_d:.9f}")

        # Print loss weights
        print(f"🔹 Loss Weights → L1: {config.L1_LOSS_WEIGHT}, Adv: {config.ADV_LOSS_WEIGHT}, FM: {config.FM_LOSS_WEIGHT}")

        ###### 🔹 Training Phase ###### 
        for batch_idx, batch in enumerate(train_dataloader):
            input_edges, gt_edges, mask, gray = (
                batch["input_edge"].to(config.DEVICE),
                batch["gt_edge"].to(config.DEVICE),
                batch["mask"].to(config.DEVICE),
                batch["gray"].to(config.DEVICE)
                )


            ###### 🔹 Train Generator (G1) ###### 
            g1.train()
            optimizer_g.zero_grad()
            with torch.amp.autocast(config.DEVICE):  
                pred_edge = g1(input_edges, mask, gray)  

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
            
            # Update EMA model after each generator update
            g1_ema.update()

            ###### 🔹 Train Discriminator (D1) ###### 
            optimizer_d.zero_grad()

            with torch.amp.autocast(config.DEVICE):  
                real_pred = d1(input_edges, gt_edges)  
                fake_pred_detached = d1(input_edges, pred_edge_detached)  # Use the detached tensor

                target_fake = torch.zeros_like(fake_pred_detached, device=config.DEVICE) + 0.1
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
            if (batch_idx + 1) % config.BATCH_SAMPLING_SIZE == 0:
                print(f"  🔹 Batch [{batch_idx+1}/{len(train_dataloader)}] - G1 Loss: {loss_g.item():.4f}, D1 Loss: {loss_d.item():.4f}")

                print(f"\n📸 Saving Training Samples for batch {batch_idx+1}...\n")
                # Apply EMA for sample generation
                g1_ema.apply_shadow()
                with torch.no_grad():  # Add torch.no_grad() here for consistency
                    pred_edge_ema = g1(input_edges, mask, gray)
                # Use epoch, not epoch+1 for consistent numbering
                save_generated_images(epoch, input_edges, mask, gt_edges, gray, pred_edge_ema, mode="train", batch_idx=batch_idx+1)
                g1_ema.restore()

        # Step the learning rate scheduler
        scheduler_g.step()
        scheduler_d.step()

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
        save_losses_to_json(batch_losses, epoch_losses, config.LOSS_PLOT_DIR_G1)
        
        # Reset batch losses for the next epoch to avoid duplication
        batch_losses = {'batch_idx': [], 'G1_L1': [], 'G1_Adv': [], 'G1_FM': [], 'D1_Real': [], 'D1_Fake': []}

        # Then plot using the saved JSON files
        plot_losses(config.LOSS_PLOT_DIR_G1)

        # Save best model checkpoint if G1 loss improves
        if avg_g1_loss < best_g1_loss:
            best_g1_loss = avg_g1_loss
            # Apply EMA weights for saving the best model
            g1_ema.apply_shadow()
            # Don't pass the empty batch_losses here
            save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_g1_loss, history, batch_losses, epoch_losses, g1_ema)
            g1_ema.restore()
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            epochs_no_improve += 1
            
        # **Save Training Samples Every Epoch**
        if (epoch) % config.TRAINING_SAMPLE_EPOCHS == 0:
            print(f"\n📸 Saving Training Samples for Epoch {epoch}...\n")
            # Apply EMA weights for visualization
            g1_ema.apply_shadow()
            with torch.no_grad():
                pred_edge_ema = g1(input_edges, mask, gray)
                save_generated_images(
                epoch=epoch, 
                input_edges=input_edges, 
                gt_edges=gt_edges, 
                pred_edges=pred_edge_ema, 
                masks=mask,
                gray=gray,
                mode="train"
            )
            g1_ema.restore()

        ###### 🔹 Validation Phase ###### 
        if (epoch) % config.VALIDATION_SAMPLE_EPOCHS == 0:
            print(f"\n🔍 Running Validation for Epoch {epoch}...\n")
            g1.eval()
            # Apply EMA weights for validation
            g1_ema.apply_shadow()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_input_edges, val_gt_edges, val_mask, val_gray = (
                        val_batch["input_edge"].to(config.DEVICE),   
                        val_batch["gt_edge"].to(config.DEVICE),  
                        val_batch["mask"].to(config.DEVICE),
                        val_batch["gray"].to(config.DEVICE)
                    )

                    val_pred_edge = g1(val_input_edges, val_mask, val_gray)

                    # Save validation images
                    save_generated_images(
                        epoch=epoch, 
                        input_edges=val_input_edges, 
                        gt_edges=val_gt_edges, 
                        pred_edges=val_pred_edge, 
                        masks=val_mask,
                        gray = val_gray,
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

        # Calculate and print model hash
        model_hash = calculate_model_hash(g1)
        print(f"Model hash after epoch {epoch}: {model_hash}")

    print(f"\n✅ Training Completed in {time.time() - start_time:.2f} seconds.\n")
