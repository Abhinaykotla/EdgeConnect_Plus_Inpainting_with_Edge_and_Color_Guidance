# train_loops.py (G2 training extension)

import time
import torch
import torch.nn.functional as F
from dataloader_g2 import get_dataloader_g2
from g2_model import InpaintingGeneratorG2, InpaintDiscriminatorG2
from loss_functions import adversarial_loss, l1_loss, perceptual_loss, style_loss, feature_matching_loss,VGG16FeatureExtractor
from utils_g2 import (
    save_checkpoint_g2, 
    load_checkpoint_g2, 
    plot_losses_g2, 
    save_generated_images_g2, 
    save_losses_to_json_g2,
    print_model_info_g2
)
from config import config


def calculate_model_hash_g2(model):
    """Calculate a hash of model parameters to track changes"""
    params = [p.data for p in model.parameters()]
    param_shapes = [p.shape for p in params]
    param_data = torch.cat([p.flatten() for p in params if p.numel() > 0])
    param_hash = hash(str(param_shapes) + str(param_data.sum().item()) + str(param_data[:5].tolist()))
    return param_hash


class EMA:
    """Exponential Moving Average for model weights."""
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
        """Apply EMA weights for inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters after inference"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def gradient_penalty(discriminator, input_img, guidance_img, real_img, fake_img):
    alpha = torch.rand(real_img.size(0), 1, 1, 1, device=config.DEVICE)
    interpolated = (alpha * real_img + (1 - alpha) * fake_img).detach().requires_grad_(True)

    d_interpolated = discriminator(input_img.detach(), interpolated)

    grad_outputs = torch.ones_like(d_interpolated, device=config.DEVICE)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_g2_and_d2():
    print("\n🔹 Initializing G2 & D2 Model Training Setup...\n")

    # Model Init
    g2 = InpaintingGeneratorG2().to(config.DEVICE)
    d2 = InpaintDiscriminatorG2().to(config.DEVICE)
    g2_ema = EMA(g2, decay=0.999)  # EMA model
    vgg = VGG16FeatureExtractor().to(config.DEVICE).eval()

    optimizer_g = torch.optim.Adam(g2.parameters(), lr=config.LEARNING_RATE_G2,
                                   betas=(config.BETA1_G2, config.BETA2_G2), weight_decay=config.WEIGHT_DECAY_G2)
    optimizer_d = torch.optim.Adam(d2.parameters(), lr=config.LEARNING_RATE_G2 * config.D2G_LR_RATIO_G2,
                                   betas=(config.BETA1_G2, config.BETA2_G2), weight_decay=config.WEIGHT_DECAY_G2)

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.9)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.9)

    # Use Mixed Precision for Faster Training
    scaler = torch.amp.GradScaler(device=config.DEVICE)

    print("Loading data into Dataloaders")
    # Load datasets
    train_loader = get_dataloader_g2("train")
    val_loader = get_dataloader_g2("val")

    best_loss = float("inf")
    history = {"g2_loss": [], "d2_loss": []}
    batch_losses = {
        'batch_idx': [], 
        'G2_L1': [], 
        'G2_Adv': [], 
        'G2_FM': [],  # Add FM loss tracking
        'G2_Perc': [], 
        'G2_Style': [], 
        'D2_Real': [], 
        'D2_Fake': [], 
        'D2_GP': []
    }
    epoch_losses = {'epoch': [], 'G2_Loss': [], 'D2_Loss': []}

    # Training Loop
    num_epochs = config.EPOCHS
    print(f"🔹 Training for a max of {num_epochs} Epochs on {config.DEVICE} with early stopping patience of {config.EARLY_STOP_PATIENCE} ...\n")

    # Print loss weights
    print(f"🔹 Loss Weights → L1: {config.L1_LOSS_WEIGHT_G2}, Adv: {config.ADV_LOSS_WEIGHT_G2}, " 
          f"FM: {config.FM_LOSS_WEIGHT_G2}, Perceptual: {config.PERCEPTUAL_LOSS_G2}, Style: {config.STYLE_LOSS_WEIGHT_G2}")

    print("🔹 Checking for old checkpoints\n")
    print("Model Hash before loading:", calculate_model_hash_g2(g2))

    # Load checkpoint if available
    start_epoch, best_loss, history, batch_losses, epoch_losses = load_checkpoint_g2(g2, d2, optimizer_g, optimizer_d, g2_ema)

    print("Model Hash after loading:", calculate_model_hash_g2(g2))

    # Early Stopping Parameters
    patience = config.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    start_time = time.time()

    print_model_info_g2(g2, model_name="Inpainting Generator (G2)")
    print_model_info_g2(d2, model_name="Discriminator (D2)")

    for epoch in range(start_epoch, config.EPOCHS + 1):
        g2.train(); d2.train()
        epoch_g_loss, epoch_d_loss = 0, 0

        for batch_idx, batch in enumerate(train_loader):
            input_img = batch["input_img"].to(config.DEVICE)
            guidance_img = batch["guidance_img"].to(config.DEVICE)
            mask = batch["mask"].to(config.DEVICE)
            gt_img = batch["gt_img"].to(config.DEVICE)

            ########################### G2 TRAINING ###########################
            optimizer_g.zero_grad()
            with torch.amp.autocast(config.DEVICE):
                pred_img = g2(input_img, guidance_img, mask)

                # Calculate G2 losses
                g2_l1 = l1_loss(pred_img, gt_img) * config.L1_LOSS_WEIGHT_G2
                g2_adv = adversarial_loss(d2(input_img, pred_img), torch.ones_like(d2(input_img, pred_img))) * config.ADV_LOSS_WEIGHT_G2
                g2_perc = perceptual_loss(vgg, pred_img, gt_img) * config.PERCEPTUAL_LOSS_G2
                g2_style = style_loss(vgg, pred_img, gt_img) * config.STYLE_LOSS_WEIGHT_G2

                # Feature Matching Loss - Fixed implementation
                # Extract features from discriminator for real and fake images
                real_features = d2.get_features(input_img, gt_img)
                fake_features = d2.get_features(input_img, pred_img)
                g2_fm = feature_matching_loss(real_features, fake_features) * config.FM_LOSS_WEIGHT_G2

                # Total G2 loss
                g_loss = g2_l1 + g2_adv + g2_fm + g2_perc + g2_style

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # Update EMA model
            g2_ema.update()

            ########################### D2 TRAINING ###########################
            optimizer_d.zero_grad()
            with torch.amp.autocast(config.DEVICE):
                d_real = d2(input_img, gt_img)
                d_fake = d2(input_img, pred_img.detach())

                d2_real_loss = adversarial_loss(d_real, torch.ones_like(d_real))
                d2_fake_loss = adversarial_loss(d_fake, torch.zeros_like(d_fake))
                gp = gradient_penalty(d2, input_img, guidance_img, gt_img, pred_img.detach())
                d_loss = (d2_real_loss + d2_fake_loss) * 0.5 + config.GRADIENT_PENALTY_WEIGHT_G2 * gp

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Logging
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            batch_losses['batch_idx'].append(batch_idx)
            batch_losses['G2_L1'].append(g2_l1.item())
            batch_losses['G2_Adv'].append(g2_adv.item())
            batch_losses['G2_FM'].append(g2_fm.item())
            batch_losses['G2_Perc'].append(g2_perc.item())
            batch_losses['G2_Style'].append(g2_style.item())
            batch_losses['D2_Real'].append(d2_real_loss.item())
            batch_losses['D2_Fake'].append(d2_fake_loss.item())
            
            if 'D2_GP' not in batch_losses:
                batch_losses['D2_GP'] = []
            batch_losses['D2_GP'].append(gp.item())

            # Add within batch loop in G2:
            if (batch_idx + 1) % config.BATCH_SAMPLING_SIZE == 0:
                print(f"  🔹 Batch [{batch_idx+1}/{len(train_loader)}] - G2 Loss: {g_loss.item():.4f}, D2 Loss: {d_loss.item():.4f}")
                
                print(f"\n📸 Saving Training Samples for batch {batch_idx+1}...\n")
                # Apply EMA for sample generation
                g2_ema.apply_shadow()
                g2.eval()
                with torch.no_grad():
                    pred_img_ema = g2(input_img, guidance_img, mask)
                save_generated_images_g2(epoch, input_img, guidance_img, mask, gt_img, pred_img_ema, mode="train", batch_idx=batch_idx+1)
                g2_ema.restore()

        scheduler_g.step()
        scheduler_d.step()

        # Epoch logging
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        epoch_losses['epoch'].append(epoch)
        epoch_losses['G2_Loss'].append(avg_g_loss)
        epoch_losses['D2_Loss'].append(avg_d_loss)

        history["g2_loss"].append(avg_g_loss)
        history["d2_loss"].append(avg_d_loss)

        save_losses_to_json_g2(batch_losses, epoch_losses, config.LOSS_PLOT_DIR_G2)

        batch_losses = {'batch_idx': [], 'G2_L1': [], 'G2_Adv': [], 'G2_FM': [], 'G2_Perc': [], 'G2_Style': [], 'D2_Real': [], 'D2_Fake': [], 'D2_GP': []}
        
        plot_losses_g2(config.LOSS_PLOT_DIR_G2)


        if avg_g_loss < best_loss:
            best_loss = avg_g_loss
            save_checkpoint_g2(epoch, g2, d2, optimizer_g, optimizer_d, best_loss, history, batch_losses, epoch_losses, g2_ema)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered after {epoch} epochs without improvement")
                break

        if epoch % config.TRAINING_SAMPLE_EPOCHS == 0:
            print(f"📸 Saving Training Samples at Epoch {epoch}...")
            g2_ema.apply_shadow()
            g2.eval()
            with torch.no_grad():
                samples = next(iter(train_loader))
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
            g2_ema.restore()

        if epoch % config.VALIDATION_SAMPLE_EPOCHS == 0:
            print(f"🔍 Validation Samples at Epoch {epoch}...")
            g2_ema.apply_shadow()
            g2.eval()
            with torch.no_grad():
                val_samples = next(iter(val_loader))
                pred_img = g2(val_samples["input_img"].to(config.DEVICE),
                              val_samples["guidance_img"].to(config.DEVICE),
                              val_samples["mask"].to(config.DEVICE))
                save_generated_images_g2(epoch, val_samples["input_img"], val_samples["mask"], val_samples["gt_img"],
                                         val_samples["guidance_img"], pred_img, mode="val")
            g2_ema.restore()

        elapsed_time = time.time() - start_time
        print(f"✅ Epoch {epoch}/{num_epochs} - G2 Loss: {avg_g_loss:.4f}, D2 Loss: {avg_d_loss:.4f}, Time: {elapsed_time/60:.2f} min\n")
