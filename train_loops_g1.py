# train_loops_g1.py - Training routines for EdgeConnect+ G1 edge generator model
# Implements a GAN-based training approach with adversarial, L1, feature matching, and perceptual losses

import time
import torch
from dataloader_g1 import get_dataloader_g1
from loss_functions import adversarial_loss, l1_loss, feature_matching_loss
from g1_model import EdgeGenerator, EdgeDiscriminator
from utils_g1 import save_checkpoint, load_checkpoint, save_losses_to_json, plot_losses, save_generated_images, print_model_info, calculate_model_hash
from config import config
from loss_functions import VGG16FeatureExtractor, perceptual_loss

# Initialize VGG feature extractor for perceptual loss calculation
vgg = VGG16FeatureExtractor().to(config.DEVICE).eval()

class EMA:
    """
    Exponential Moving Average for model weights.
    
    Maintains a moving average of model parameters to produce more stable results,
    particularly useful for reducing variance in generative models.
    
    Args:
        model (torch.nn.Module): Model to apply EMA to
        decay (float): Decay rate for the moving average (default: 0.999)
                    Higher values give more weight to past parameters
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}  # Dictionary to store EMA parameters
        self.backup = {}  # Dictionary to backup original parameters
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """
        Update EMA parameters after each optimization step.
        
        Applies the exponential moving average formula to each parameter:
        EMA_param = decay * EMA_param + (1 - decay) * current_param
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """
        Apply EMA parameters to the model for inference.
        
        Stores a backup of the current parameters and replaces them with EMA parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """
        Restore original parameters to the model after inference.
        
        Replaces EMA parameters with the original backed-up parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def gradient_penalty(discriminator, input_edges, gray, real_edge, fake_edge):
    """
    Compute gradient penalty for WGAN-GP training stability.
    
    Creates interpolated samples between real and fake data, then calculates 
    penalty based on gradient norm deviation from 1.
    
    Args:
        discriminator (torch.nn.Module): The discriminator model
        input_edges (torch.Tensor): Input edge map of shape [B, 1, H, W]
        gray (torch.Tensor): Grayscale image of shape [B, 1, H, W]
        real_edge (torch.Tensor): Real edge map of shape [B, 1, H, W]
        fake_edge (torch.Tensor): Fake/generated edge map of shape [B, 1, H, W]
        
    Returns:
        torch.Tensor: Scalar gradient penalty loss
    """
    # Sample random points between real and fake samples
    alpha = torch.rand(real_edge.size(0), 1, 1, 1, device=config.DEVICE)
    interpolated = (alpha * real_edge + (1 - alpha) * fake_edge).detach().requires_grad_(True)

    # Calculate discriminator output for interpolated points
    d_interpolates = discriminator(input_edges.detach(), gray.detach(), interpolated)

    # Prepare gradient outputs (all ones)
    grad_outputs = torch.ones_like(d_interpolates, device=config.DEVICE)

    # Get gradients w.r.t. interpolated points
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Calculate gradient penalty: (||grad|| - 1)^2
    gradients = gradients.view(gradients.size(0), -1)  # Flatten gradients
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # Mean squared deviation from norm 1
    return gp


# Training Loop for EdgeConnect+ G1 Model
def train_g1_and_d1():
    """
    Main training loop for the EdgeConnect+ edge generation model.
    
    Trains the EdgeGenerator (G1) and EdgeDiscriminator (D1) models using GAN approach with
    L1 loss, adversarial loss, feature matching loss, and VGG perceptual loss.
    Implements EMA, mixed precision training, learning rate scheduling, and early stopping.
    
    Returns:
        None: Results are saved as model checkpoints, loss plots, and sample images
    """

    print("\nINFO: Initializing Model & Training Setup...\n")

    # Initialize models
    g1 = EdgeGenerator().to(config.DEVICE)
    d1 = EdgeDiscriminator().to(config.DEVICE)
    
    # Initialize EMA for G1 model with a decay rate of 0.999
    # EMA maintains a moving average of parameters for more stable results
    g1_ema = EMA(g1, decay=0.999)

    # Optimizers using config settings
    optimizer_g = torch.optim.Adam(
        g1.parameters(), 
        lr=config.LEARNING_RATE_G1, 
        betas=(config.BETA1, config.BETA2), 
        weight_decay=config.WEIGHT_DECAY
    )

    # D1 typically uses a scaled learning rate compared to G1
    optimizer_d = torch.optim.Adam(
        d1.parameters(), 
        lr=config.LEARNING_RATE_G1 * config.D2G_LR_RATIO_G1,  
        betas=(config.BETA1, config.BETA2), 
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler - reduces learning rate every 10 epochs
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.85)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.85)

    # Use Mixed Precision for Faster Training with reduced memory usage
    scaler = torch.amp.GradScaler(device=config.DEVICE)

    print("INFO: Loading data into Dataloaders")
    # Load datasets for training and validation
    train_dataloader = get_dataloader_g1(split="train", use_mask=True)
    val_dataloader = get_dataloader_g1(split="val", use_mask=True)  

    # Training Loop configuration
    num_epochs = config.EPOCHS
    print(f"INFO: Training for a max of {num_epochs} Epochs on {config.DEVICE} with early stopping patience of {config.EARLY_STOP_PATIENCE} ...\n")

    # Print loss weights - inform user about the importance of each loss component
    print(f"INFO: Loss Weights → L1: {config.L1_LOSS_WEIGHT}, Adv: {config.ADV_LOSS_WEIGHT}, FM: {config.FM_LOSS_WEIGHT}")

    print("INFO: Checking for old checkpoints\n")
    print("Model Hash before loading:", calculate_model_hash(g1)) 

    # Load checkpoint if available - resumes training from last saved state
    start_epoch, best_g1_loss, history, batch_losses, epoch_losses = load_checkpoint(g1, d1, optimizer_g, optimizer_d, g1_ema)

    print("Model Hash after loading:", calculate_model_hash(g1)) 

    # Early Stopping Parameters - prevents overfitting by monitoring validation loss
    patience = config.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    start_time = time.time()

    # Print model structure and parameter information
    print_model_info(g1, model_name="Generator (G1)")
    print_model_info(d1, model_name="Discriminator (D1)")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        total_g_loss = 0.0
        total_d_loss = 0.0

        # Print current learning rates for monitoring training progress
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        print(f"INFO: Current Learning Rates → G1: {current_lr_g:.9f}, D1: {current_lr_d:.9f}")

        # Print loss weights - remind user of current configuration
        print(f"INFO: Loss Weights → L1: {config.L1_LOSS_WEIGHT}, Adv: {config.ADV_LOSS_WEIGHT}, FM: {config.FM_LOSS_WEIGHT}")

        ###### Training Phase ###### 
        for batch_idx, batch in enumerate(train_dataloader):
            # Extract and move tensors to device
            # Shape [B, 1, H, W] for each tensor
            input_edges, gt_edges, mask, gray = (
                batch["input_edge"].to(config.DEVICE),
                batch["gt_edge"].to(config.DEVICE),
                batch["mask"].to(config.DEVICE),
                batch["gray"].to(config.DEVICE)
                )


            ###### Train Generator (G1) ###### 
            g1.train()
            optimizer_g.zero_grad()
            with torch.amp.autocast(config.DEVICE):  # Mixed precision for faster training
                # Forward pass through G1
                pred_edge = g1(input_edges, mask, gray)  # Shape: [B, 1, H, W]

                # L1 Loss - pixel-wise reconstruction accuracy
                g1_loss_l1 = l1_loss(pred_edge, gt_edges) * config.L1_LOSS_WEIGHT  

                # Store detached predictions for discriminator step
                # Detaching prevents backprop through generator when training discriminator
                pred_edge_detached = pred_edge.detach()
                
                # Adversarial Loss - based on discriminator's ability to distinguish fake from real
                fake_pred = d1(input_edges, gray, pred_edge)
                # Label smoothing (0.9 instead of 1.0) for more stable GAN training
                target_real = torch.ones_like(fake_pred, device=config.DEVICE) * 0.9  
                g1_loss_adv = adversarial_loss(fake_pred, target_real)  

                # Feature Matching Loss - compares intermediate discriminator features
                real_features = d1(input_edges, gray, gt_edges).detach()  # Real edge features from D1
                g1_loss_fm = feature_matching_loss(real_features, fake_pred) * config.FM_LOSS_WEIGHT  

                # VGG Perceptual Loss - compare features in VGG space
                # Repeat single channel to create 3-channel input for VGG
                g1_loss_perc = perceptual_loss(vgg, pred_edge.repeat(1, 3, 1, 1), gt_edges.repeat(1, 3, 1, 1)) * config.VGG_LOSS_WEIGHT

                # Total Generator Loss - weighted combination of all loss components
                loss_g = g1_loss_l1 + g1_loss_adv + g1_loss_fm + g1_loss_perc

            # Mixed precision backward pass and optimization
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()
            
            # Update EMA model after each generator update
            g1_ema.update()

            ###### Train Discriminator (D1) ###### 
            optimizer_d.zero_grad()

            with torch.amp.autocast(config.DEVICE):  # Mixed precision for faster training
                # Get discriminator predictions for real edges
                real_pred = d1(input_edges, gray, gt_edges)  
                # Get discriminator predictions for fake edges (using detached predictions)
                fake_pred_detached = d1(input_edges, gray, pred_edge_detached)

                # Label smoothing for discriminator (0.9 for real, 0.1 for fake)
                # Helps prevent overconfidence and improve stability
                target_fake = torch.zeros_like(fake_pred_detached, device=config.DEVICE) + 0.1
                real_loss = adversarial_loss(real_pred, target_real)
                fake_loss = adversarial_loss(fake_pred_detached, target_fake)

                # Gradient penalty for WGAN-GP style training
                lambda_gp = 10  # Gradient penalty weight
                gp = gradient_penalty(d1, input_edges, gray, gt_edges.detach(), pred_edge_detached)
                # Total discriminator loss combines real/fake classification with gradient penalty
                loss_d = (real_loss + fake_loss) / 2 + lambda_gp * gp

            # Mixed precision backward pass and optimization for discriminator
            scaler.scale(loss_d).backward(retain_graph=True)
            scaler.step(optimizer_d)
            scaler.update()

            # Track Losses for plotting and analysis
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            batch_losses['batch_idx'].append(batch_idx)
            batch_losses['G1_L1'].append(g1_loss_l1.item())
            batch_losses['G1_Adv'].append(g1_loss_adv.item())
            batch_losses['G1_FM'].append(g1_loss_fm.item())
            batch_losses['D1_Real'].append(real_loss.item())
            batch_losses['D1_Fake'].append(fake_loss.item())
            batch_losses.setdefault('G1_VGG', []).append(g1_loss_perc.item())

            # Print progress every N batches as defined in config
            if (batch_idx + 1) % config.BATCH_SAMPLING_SIZE == 0:
                print(f"  INFO: Batch [{batch_idx+1}/{len(train_dataloader)}] - G1 Loss: {loss_g.item():.4f}, D1 Loss: {loss_d.item():.4f}")

                print(f"\nINFO: Saving Training Samples for batch {batch_idx+1}...\n")
                # Apply EMA for sample generation - more stable outputs
                g1_ema.apply_shadow()
                with torch.no_grad():  # No gradient computation for inference
                    pred_edge_ema = g1(input_edges, mask, gray)
                # Use epoch, not epoch+1 for consistent numbering
                save_generated_images(epoch, input_edges, mask, gt_edges, gray, pred_edge_ema, mode="train", batch_idx=batch_idx+1)
                g1_ema.restore()

        # Step the learning rate scheduler - gradually reduce LR over epochs
        scheduler_g.step()
        scheduler_d.step()

        # Compute average loss for the epoch
        avg_g1_loss = total_g_loss / len(train_dataloader)
        avg_d1_loss = total_d_loss / len(train_dataloader)
        epoch_losses['epoch'].append(epoch)
        epoch_losses['G1_Loss'].append(avg_g1_loss)
        epoch_losses['D1_Loss'].append(avg_d1_loss)

        # Save training history for plotting and analysis
        history["g1_loss"].append(avg_g1_loss)
        history["d1_loss"].append(avg_d1_loss)

        # First save the current losses to JSON files for persistence
        save_losses_to_json(batch_losses, epoch_losses, config.LOSS_PLOT_DIR_G1)
        
        # Reset batch losses for the next epoch to avoid duplication
        batch_losses = {
                'batch_idx': [],
                'G1_L1': [],
                'G1_Adv': [],
                'G1_FM': [],
                'G1_VGG': [],
                'D1_Real': [],
                'D1_Fake': []
            }

        # Then plot using the saved JSON files - creates visualization of training progress
        plot_losses(config.LOSS_PLOT_DIR_G1)

        # Save best model checkpoint if G1 loss improves
        if avg_g1_loss < best_g1_loss:
            best_g1_loss = avg_g1_loss
            # Apply EMA weights for saving the best model - more stable parameters
            g1_ema.apply_shadow()
            # Don't pass the empty batch_losses here
            save_checkpoint(epoch, g1, d1, optimizer_g, optimizer_d, best_g1_loss, history, batch_losses, epoch_losses, g1_ema)
            g1_ema.restore()
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            epochs_no_improve += 1
            
        # Save Training Samples Every N Epochs as configured
        if (epoch) % config.TRAINING_SAMPLE_EPOCHS == 0:
            print(f"\nINFO: Saving Training Samples for Epoch {epoch}...\n")
            # Apply EMA weights for visualization - more stable outputs
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

        ###### Validation Phase ###### 
        if (epoch) % config.VALIDATION_SAMPLE_EPOCHS == 0:
            print(f"\nINFO: Running Validation for Epoch {epoch}...\n")
            g1.eval()  # Set model to evaluation mode
            # Apply EMA weights for validation - more stable outputs
            g1_ema.apply_shadow()
            with torch.no_grad():  # No gradient computation needed for validation
                for val_batch in val_dataloader:
                    val_input_edges, val_gt_edges, val_mask, val_gray = (
                        val_batch["input_edge"].to(config.DEVICE),   
                        val_batch["gt_edge"].to(config.DEVICE),  
                        val_batch["mask"].to(config.DEVICE),
                        val_batch["gray"].to(config.DEVICE)
                    )

                    val_pred_edge = g1(val_input_edges, val_mask, val_gray)

                    # Save validation images for visual inspection of model performance
                    save_generated_images(
                        epoch=epoch, 
                        input_edges=val_input_edges, 
                        gt_edges=val_gt_edges, 
                        pred_edges=val_pred_edge, 
                        masks=val_mask,
                        gray=val_gray,
                        mode="val"
                    )
                    break  # Process only 1 batch per validation to save time
            g1_ema.restore()

        # Early stopping check - prevent overfitting by stopping when no improvement
        if epochs_no_improve >= patience:
            print(f"\nSTOPPING: Early stopping triggered after {patience} epochs without improvement.")
            break

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"\nINFO: Epoch [{epoch}/{num_epochs}] Completed in {epoch_duration:.2f}s - G1 Loss: {avg_g1_loss:.4f}, D1 Loss: {avg_d1_loss:.4f}\n")

        # Calculate and print model hash - useful for detecting unintended parameter changes
        model_hash = calculate_model_hash(g1)
        print(f"Model hash after epoch {epoch}: {model_hash}")

    print(f"\nCOMPLETE: Training Completed in {time.time() - start_time:.2f} seconds.\n")


if __name__ == '__main__':
    import multiprocessing
    
    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()

    print("INFO: Starting EdgeConnect G1 (Edge Generator) training...")
    train_g1_and_d1()