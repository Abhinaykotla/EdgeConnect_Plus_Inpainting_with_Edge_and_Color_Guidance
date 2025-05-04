# train_loops.py (G2 training extension)
# Implements training procedures for the second generator (G2) of EdgeConnect+
# G2 generates inpainted images guided by edge maps and color information

import time
import torch
import torch.nn.functional as F
from dataloader_g2 import get_dataloader_g2
from g2_model import InpaintingGeneratorG2, InpaintDiscriminatorG2
from loss_functions import adversarial_loss, l1_loss, perceptual_loss, style_loss, feature_matching_loss, VGG16FeatureExtractor
from utils_g2 import (
    save_checkpoint_g2, 
    load_checkpoint_g2, 
    plot_losses_g2, 
    save_generated_images_g2, 
    save_losses_to_json_g2,
    print_model_info_g2,
    calculate_model_hash_g2
)
from config import config


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


def gradient_penalty(discriminator, input_img, guidance_img, real_img, fake_img):
    """
    Compute gradient penalty for WGAN-GP training stability.
    
    Creates interpolated samples between real and fake data, then calculates 
    penalty based on gradient norm deviation from 1.
    
    Args:
        discriminator (torch.nn.Module): The discriminator model
        input_img (torch.Tensor): Input masked image of shape [B, 3, H, W]
        guidance_img (torch.Tensor): Guidance image of shape [B, 3, H, W]
        real_img (torch.Tensor): Real ground truth image of shape [B, 3, H, W]
        fake_img (torch.Tensor): Fake/generated image of shape [B, 3, H, W]
        
    Returns:
        torch.Tensor: Scalar gradient penalty loss
    """
    # Sample random points between real and fake samples
    alpha = torch.rand(real_img.size(0), 1, 1, 1, device=config.DEVICE)
    interpolated = (alpha * real_img + (1 - alpha) * fake_img).detach().requires_grad_(True)

    # Calculate discriminator output for interpolated points
    d_interpolated = discriminator(input_img.detach(), interpolated)

    # Prepare gradient outputs (all ones)
    grad_outputs = torch.ones_like(d_interpolated, device=config.DEVICE)

    # Get gradients w.r.t. interpolated points
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
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


def train_g2_and_d2():
    """
    Main training function for the EdgeConnect+ G2 inpainting model.
    
    Trains the InpaintingGeneratorG2 (G2) and InpaintDiscriminatorG2 (D2) models using a GAN approach
    with multiple loss components including L1, adversarial, feature matching, perceptual and style losses.
    Implements EMA, mixed precision training, and early stopping.
    
    Returns:
        None: Results are saved as checkpoints, loss plots, and sample images
    """
    print("\nINFO: Initializing G2 & D2 Model Training Setup...\n")

    # Model Initialization
    g2 = InpaintingGeneratorG2().to(config.DEVICE)
    d2 = InpaintDiscriminatorG2().to(config.DEVICE)
    g2_ema = EMA(g2, decay=0.999)  # EMA model for stable outputs
    vgg = VGG16FeatureExtractor().to(config.DEVICE).eval()  # Frozen VGG for perceptual/style losses

    # Optimizers with hyperparameters from config
    optimizer_g = torch.optim.Adam(g2.parameters(), lr=config.LEARNING_RATE_G2,
                                   betas=(config.BETA1_G2, config.BETA2_G2), weight_decay=config.WEIGHT_DECAY_G2)
    optimizer_d = torch.optim.Adam(d2.parameters(), lr=config.LEARNING_RATE_G2 * config.D2G_LR_RATIO_G2,
                                   betas=(config.BETA1_G2, config.BETA2_G2), weight_decay=config.WEIGHT_DECAY_G2)

    # Learning rate schedulers - reduce LR when training plateaus
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=5, gamma=0.9)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=5, gamma=0.9)

    # Use Mixed Precision for Faster Training (AMP)
    scaler = torch.amp.GradScaler(device=config.DEVICE)

    print("INFO: Loading data into Dataloaders")
    # Load datasets for training and validation
    train_loader = get_dataloader_g2("train")
    val_loader = get_dataloader_g2("val")

    # Training tracking variables
    best_loss = float("inf")
    history = {"g2_loss": [], "d2_loss": []}
    batch_losses = {
        'batch_idx': [], 
        'G2_L1': [], 
        'G2_Adv': [], 
        'G2_FM': [],  # Feature matching loss tracking
        'G2_Perc': [],  # Perceptual loss tracking
        'G2_Style': [],  # Style loss tracking
        'D2_Real': [], 
        'D2_Fake': [], 
        'D2_GP': []  # Gradient penalty tracking
    }
    epoch_losses = {'epoch': [], 'G2_Loss': [], 'D2_Loss': []}

    # Training Loop configuration
    num_epochs = config.EPOCHS
    print(f"INFO: Training for a max of {num_epochs} Epochs on {config.DEVICE} with early stopping patience of {config.EARLY_STOP_PATIENCE} ...\n")

    # Print loss weights for reference
    print(f"INFO: Loss Weights → L1: {config.L1_LOSS_WEIGHT_G2}, Adv: {config.ADV_LOSS_WEIGHT_G2}, " 
          f"FM: {config.FM_LOSS_WEIGHT_G2}, Perceptual: {config.PERCEPTUAL_LOSS_G2}, Style: {config.STYLE_LOSS_WEIGHT_G2}")

    print("INFO: Checking for old checkpoints\n")
    print("Model Hash before loading:", calculate_model_hash_g2(g2))

    # Load checkpoint if available - resumes training from last saved state
    start_epoch, best_loss, history, batch_losses, epoch_losses = load_checkpoint_g2(g2, d2, optimizer_g, optimizer_d, g2_ema)

    print("Model Hash after loading:", calculate_model_hash_g2(g2))

    # Early Stopping Parameters
    patience = config.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    # Track overall training time
    start_time = time.time()

    # Print model architecture information
    print_model_info_g2(g2, model_name="Inpainting Generator (G2)")
    print_model_info_g2(d2, model_name="Discriminator (D2)")

    # Main training loop
    for epoch in range(start_epoch, config.EPOCHS + 1):
        g2.train(); d2.train()  # Set models to training mode
        epoch_g_loss, epoch_d_loss = 0, 0  # Track epoch-level losses

        # Process each batch in the training loader
        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to the device
            input_img = batch["input_img"].to(config.DEVICE)           # [B, 3, H, W] - RGB image with holes
            guidance_img = batch["guidance_img"].to(config.DEVICE)      # [B, 3, H, W] - Edge and color guidance
            mask = batch["mask"].to(config.DEVICE)                      # [B, 1, H, W] - Binary mask (1=valid, 0=hole)
            gt_img = batch["gt_img"].to(config.DEVICE)                  # [B, 3, H, W] - Ground truth image

            ########################### G2 TRAINING ###########################
            optimizer_g.zero_grad()  # Clear previous gradients
            with torch.amp.autocast(config.DEVICE):  # Mixed precision for faster training
                # Generate inpainted image
                pred_img = g2(input_img, guidance_img, mask)  # [B, 3, H, W]

                # Calculate multiple G2 loss components
                g2_l1 = l1_loss(pred_img * mask, gt_img * mask) * config.L1_LOSS_WEIGHT_G2  # L1 only in valid regions
                g2_adv = adversarial_loss(d2(input_img, pred_img), torch.ones_like(d2(input_img, pred_img))) * config.ADV_LOSS_WEIGHT_G2
                g2_perc = perceptual_loss(vgg, pred_img, gt_img) * config.PERCEPTUAL_LOSS_G2  # VGG-based perceptual loss
                g2_style = style_loss(vgg, pred_img, gt_img) * config.STYLE_LOSS_WEIGHT_G2  # Style loss for texture matching

                # Handle potential NaN issues in style loss
                if torch.isnan(g2_style):
                    print("WARNING: NaN detected in style loss")
                    # Provide diagnostic information about inputs
                    print(f"pred_img stats: min={pred_img.min().item()}, max={pred_img.max().item()}")
                    print(f"gt_img stats: min={gt_img.min().item()}, max={gt_img.max().item()}")
                    # Replace NaN with zero to continue training
                    g2_style = torch.tensor(0.0, device=config.DEVICE)

                # Feature Matching Loss - from discriminator intermediate layers
                real_features = d2.get_features(input_img, gt_img)  # Features from D2 for real images
                fake_features = d2.get_features(input_img, pred_img)  # Features from D2 for fake images
                g2_fm = feature_matching_loss(real_features, fake_features) * config.FM_LOSS_WEIGHT_G2

                # Total G2 loss - weighted combination of all components
                g_loss = g2_l1 + g2_adv + g2_fm + g2_perc + g2_style

            # Mixed precision backward pass and optimization
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # Update EMA model after generator update
            g2_ema.update()

            ########################### D2 TRAINING ###########################
            optimizer_d.zero_grad()  # Clear previous gradients
            with torch.amp.autocast(config.DEVICE):  # Mixed precision for faster training
                # Get discriminator outputs for real and fake images
                d_real = d2(input_img, gt_img)  # [B, 1, H', W'] - Real image classification
                d_fake = d2(input_img, pred_img.detach())  # [B, 1, H', W'] - Fake image classification (detached)

                # Standard GAN losses for discriminator
                d2_real_loss = adversarial_loss(d_real, torch.ones_like(d_real))  # Real samples should be classified as 1
                d2_fake_loss = adversarial_loss(d_fake, torch.zeros_like(d_fake))  # Fake samples should be classified as 0
                
                # Calculate gradient penalty for WGAN-GP style training
                gp = gradient_penalty(d2, input_img, guidance_img, gt_img, pred_img.detach())
                
                # Total discriminator loss
                d_loss = (d2_real_loss + d2_fake_loss) * 0.5 + config.GRADIENT_PENALTY_WEIGHT_G2 * gp

            # Mixed precision backward pass and optimization for discriminator
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Log losses for tracking
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            # Store individual loss components for plotting
            batch_losses['batch_idx'].append(batch_idx)
            batch_losses['G2_L1'].append(g2_l1.item())
            batch_losses['G2_Adv'].append(g2_adv.item())
            batch_losses['G2_FM'].append(g2_fm.item())
            batch_losses['G2_Perc'].append(g2_perc.item())
            batch_losses['G2_Style'].append(g2_style.item())
            batch_losses['D2_Real'].append(d2_real_loss.item())
            batch_losses['D2_Fake'].append(d2_fake_loss.item())
            
            # Ensure D2_GP key exists (for backward compatibility)
            if 'D2_GP' not in batch_losses:
                batch_losses['D2_GP'] = []
            batch_losses['D2_GP'].append(gp.item())

            # Print progress and save samples periodically
            if (batch_idx + 1) % config.BATCH_SAMPLING_SIZE == 0:
                print(f"  INFO: Batch [{batch_idx+1}/{len(train_loader)}] - G2 Loss: {g_loss.item():.4f}, D2 Loss: {d_loss.item():.4f}")
                
                print(f"\nINFO: Saving Training Samples for batch {batch_idx+1}...\n")
                # Apply EMA for more stable sample generation
                g2_ema.apply_shadow()
                g2.eval()  # Switch to evaluation mode for inference
                with torch.no_grad():  # No gradient computation for inference
                    pred_img_ema = g2(input_img, guidance_img, mask)
                save_generated_images_g2(epoch, input_img, guidance_img, mask, gt_img, pred_img_ema, mode="train", batch_idx=batch_idx+1)
                g2_ema.restore()  # Restore original weights for continued training

        # Step learning rate schedulers at the end of each epoch
        scheduler_g.step()
        scheduler_d.step()

        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        # Record epoch-level losses
        epoch_losses['epoch'].append(epoch)
        epoch_losses['G2_Loss'].append(avg_g_loss)
        epoch_losses['D2_Loss'].append(avg_d_loss)

        # Update history for plotting
        history["g2_loss"].append(avg_g_loss)
        history["d2_loss"].append(avg_d_loss)

        # Save losses to JSON for persistence and visualization
        save_losses_to_json_g2(batch_losses, epoch_losses, config.LOSS_PLOT_DIR_G2)

        # Reset batch losses for the next epoch to avoid duplication
        batch_losses = {'batch_idx': [], 'G2_L1': [], 'G2_Adv': [], 'G2_FM': [], 'G2_Perc': [], 'G2_Style': [], 'D2_Real': [], 'D2_Fake': [], 'D2_GP': []}
        
        # Generate loss plots from saved JSON data
        plot_losses_g2(config.LOSS_PLOT_DIR_G2)

        # Save checkpoint if model improved
        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            save_checkpoint_g2(epoch, g2, d2, optimizer_g, optimizer_d, best_loss, history, batch_losses, epoch_losses, g2_ema)
            epochs_no_improve = 0  # Reset early stopping counter
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"STOPPING: Early stopping triggered after {epoch} epochs without improvement")
                break

        # Generate and save training samples periodically
        if epoch % config.TRAINING_SAMPLE_EPOCHS == 0:
            print(f"INFO: Saving Training Samples at Epoch {epoch}...")
            g2_ema.apply_shadow()  # Use EMA weights for more stable outputs
            g2.eval()  # Set model to evaluation mode
            with torch.no_grad():
                samples = next(iter(train_loader))  # Get a fresh batch of samples
                pred_img = g2(samples["input_img"].to(config.DEVICE),
                              samples["guidance_img"].to(config.DEVICE),
                              samples["mask"].to(config.DEVICE))
                save_generated_images_g2(
                    epoch, 
                    samples["input_img"],  # Input image
                    samples["guidance_img"],  # Guidance image
                    samples["mask"],  # Mask
                    samples["gt_img"],  # Ground truth
                    pred_img,  # Predicted image
                    mode="train"
                )
            g2_ema.restore()  # Restore original weights

        # Run validation and save samples periodically
        if epoch % config.VALIDATION_SAMPLE_EPOCHS == 0:
            print(f"INFO: Validation Samples at Epoch {epoch}...")
            g2_ema.apply_shadow()  # Use EMA weights for validation
            g2.eval()  # Set model to evaluation mode

            with torch.no_grad():  # No gradient computation for validation
                val_samples = next(iter(val_loader))  # Get validation samples
                pred_img = g2(
                    val_samples["input_img"].to(config.DEVICE),
                    val_samples["guidance_img"].to(config.DEVICE),
                    val_samples["mask"].to(config.DEVICE)
                )
                save_generated_images_g2(
                    epoch,
                    val_samples["input_img"],  # Input image
                    val_samples["guidance_img"],  # Guidance image 
                    val_samples["mask"],  # Mask
                    val_samples["gt_img"],  # Ground truth
                    pred_img,  # Predicted image
                    mode="val"
                )
            g2_ema.restore()  # Restore original weights

        # Print epoch summary with timing information
        elapsed_time = time.time() - start_time
        print(f"COMPLETE: Epoch {epoch}/{num_epochs} - G2 Loss: {avg_g_loss:.4f}, D2 Loss: {avg_d_loss:.4f}, Time: {elapsed_time/60:.2f} min\n")


if __name__ == '__main__':
    import multiprocessing

    # Ensure proper multiprocessing behavior on Windows
    multiprocessing.freeze_support()

    print("INFO: Starting EdgeConnect G2 (Inpainting Generator) training...")
    train_g2_and_d2()