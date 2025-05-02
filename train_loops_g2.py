# train_loops.py (G2 training extension)

import time
import torch
from dataloader_g2 import get_dataloader_g2
from g2_model import InpaintingGeneratorG2, InpaintDiscriminatorG2
from loss_functions import adversarial_loss, l1_loss, perceptual_loss, style_loss, VGG16FeatureExtractor
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


def train_g2_and_d2():
    print("\n🔹 Initializing G2 & D2 Model Training Setup...\n")

    # Model Init
    g2 = InpaintingGeneratorG2().to(config.DEVICE)
    d2 = InpaintDiscriminatorG2().to(config.DEVICE)
    g2_ema = InpaintingGeneratorG2().to(config.DEVICE)  # EMA model
    vgg = VGG16FeatureExtractor().to(config.DEVICE).eval()

    # Initialize EMA model with same weights
    for param_ema, param in zip(g2_ema.parameters(), g2.parameters()):
        param_ema.data.copy_(param.data)
    g2_ema.eval()

    optimizer_g = torch.optim.Adam(g2.parameters(), lr=config.LEARNING_RATE_G2,
                                   betas=(config.BETA1_G2, config.BETA2_G2), weight_decay=config.WEIGHT_DECAY_G2)
    optimizer_d = torch.optim.Adam(d2.parameters(), lr=config.LEARNING_RATE_G2 * config.D2G_LR_RATIO_G2,
                                   betas=(config.BETA1_G2, config.BETA2_G2), weight_decay=config.WEIGHT_DECAY_G2)

    # Use Mixed Precision for Faster Training
    scaler = torch.amp.GradScaler(device=config.DEVICE)

    print("Loading data into Dataloaders")
    # Load datasets
    train_loader = get_dataloader_g2("train")
    val_loader = get_dataloader_g2("val")

    best_loss = float("inf")
    history = {"g2_loss": [], "d2_loss": []}
    batch_losses = {'batch_idx': [], 'G2_L1': [], 'G2_Adv': [], 'G2_Perc': [], 'G2_Style': [], 'D2_Real': [], 'D2_Fake': []}
    epoch_losses = {'epoch': [], 'G2_Loss': [], 'D2_Loss': []}

    # Training Loop
    num_epochs = config.EPOCHS
    print(f"🔹 Training for a max of {num_epochs} Epochs on {config.DEVICE} with early stopping patience of {config.EARLY_STOP_PATIENCE_G2} ...\n")

    # Print loss weights
    print(f"🔹 Loss Weights → L1: {config.L1_LOSS_WEIGHT_G2}, Adv: {config.ADV_LOSS_WEIGHT_G2}, " 
          f"Perceptual: {config.PERCEPTUAL_LOSS_G2}, Style: {config.STYLE_LOSS_WEIGHT_G2}")

    print("🔹 Checking for old checkpoints\n")
    print("Model Hash before loading:", calculate_model_hash_g2(g2))

    # Load checkpoint if available
    start_epoch, best_loss, history, batch_losses, epoch_losses = load_checkpoint_g2(g2, d2, optimizer_g, optimizer_d, g2_ema)

    print("Model Hash after loading:", calculate_model_hash_g2(g2))

    # Early Stopping Parameters
    patience = config.EARLY_STOP_PATIENCE_G2
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

                g2_l1 = l1_loss(pred_img, gt_img) * config.L1_LOSS_WEIGHT_G2
                g2_adv = adversarial_loss(d2(input_img, pred_img), torch.ones_like(d2(input_img, pred_img))) * config.ADV_LOSS_WEIGHT_G2
                g2_perc = perceptual_loss(vgg, pred_img, gt_img) * config.PERCEPTUAL_LOSS_G2
                g2_style = style_loss(vgg, pred_img, gt_img) * config.STYLE_LOSS_WEIGHT_G2

                g_loss = g2_l1 + g2_adv + g2_perc + g2_style

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()

            # Update EMA model
            with torch.no_grad():
                for param_ema, param in zip(g2_ema.parameters(), g2.parameters()):
                    param_ema.data = param_ema.data * 0.999 + param.data * 0.001

            ########################### D2 TRAINING ###########################
            optimizer_d.zero_grad()
            with torch.amp.autocast(config.DEVICE):
                d_real = d2(input_img, gt_img)
                d_fake = d2(input_img, pred_img.detach())

                d2_real_loss = adversarial_loss(d_real, torch.ones_like(d_real))
                d2_fake_loss = adversarial_loss(d_fake, torch.zeros_like(d_fake))
                d_loss = (d2_real_loss + d2_fake_loss) * 0.5

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Logging
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            batch_losses['batch_idx'].append(batch_idx)
            batch_losses['G2_L1'].append(g2_l1.item())
            batch_losses['G2_Adv'].append(g2_adv.item())
            batch_losses['G2_Perc'].append(g2_perc.item())
            batch_losses['G2_Style'].append(g2_style.item())
            batch_losses['D2_Real'].append(d2_real_loss.item())
            batch_losses['D2_Fake'].append(d2_fake_loss.item())

        # Epoch logging
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        epoch_losses['epoch'].append(epoch)
        epoch_losses['G2_Loss'].append(avg_g_loss)
        epoch_losses['D2_Loss'].append(avg_d_loss)

        history["g2_loss"].append(avg_g_loss)
        history["d2_loss"].append(avg_d_loss)

        save_losses_to_json_g2(batch_losses, epoch_losses, config.LOSS_PLOT_DIR_G2)
        batch_losses = {'batch_idx': [], 'G2_L1': [], 'G2_Adv': [], 'G2_Perc': [], 'G2_Style': [], 'D2_Real': [], 'D2_Fake': []}
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
            g2.eval()
            with torch.no_grad():
                samples = next(iter(train_loader))
                pred_img = g2(samples["input_img"].to(config.DEVICE),
                              samples["guidance_img"].to(config.DEVICE),
                              samples["mask"].to(config.DEVICE))
                save_generated_images_g2(epoch, samples["input_img"], samples["mask"], samples["gt_img"],
                                         samples["guidance_img"], pred_img, mode="train")

        if epoch % config.VALIDATION_SAMPLE_EPOCHS == 0:
            print(f"🔍 Validation Samples at Epoch {epoch}...")
            g2.eval()
            with torch.no_grad():
                val_samples = next(iter(val_loader))
                pred_img = g2(val_samples["input_img"].to(config.DEVICE),
                              val_samples["guidance_img"].to(config.DEVICE),
                              val_samples["mask"].to(config.DEVICE))
                save_generated_images_g2(epoch, val_samples["input_img"], val_samples["mask"], val_samples["gt_img"],
                                         val_samples["guidance_img"], pred_img, mode="val")

        elapsed_time = time.time() - start_time
        print(f"✅ Epoch {epoch}/{num_epochs} - G2 Loss: {avg_g_loss:.4f}, D2 Loss: {avg_d_loss:.4f}, Time: {elapsed_time/60:.2f} min\n")
