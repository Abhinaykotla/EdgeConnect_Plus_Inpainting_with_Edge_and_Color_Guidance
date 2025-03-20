# g1_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with dilated convolutions.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual  # Skip connection

class EdgeGenerator(nn.Module):
    """
    Edge Generator (G1) for EdgeConnect.
    Uses residual blocks and dilated convolutions for edge prediction.
    """
    def __init__(self):
        super(EdgeGenerator, self).__init__()

        # Initial Convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final Output
        self.final_conv = nn.Conv2d(64, 1, kernel_size=7, padding=3)
        self.activation = nn.Sigmoid()  # Normalized output

    def forward(self, img, mask):
        x = torch.cat((img, mask), dim=1)  # Concatenate image and mask
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final_conv(x)
        return self.activation(x)


class EdgeDiscriminator(nn.Module):
    """
    PatchGAN-based Discriminator (D1) for Edge Generation.
    Takes (input_edges, gt_edges/pred_edge) and classifies 30x30 patches as real or fake.
    """
    def __init__(self, in_channels=2):  # (input_edges + gt_edges or pred_edge)
        super(EdgeDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2):
            """
            Defines a single discriminator block with:
            - Spectral Normalization
            - Strided Convolutions for downsampling
            - LeakyReLU activation
            """
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # Define convolutional layers
        self.model = nn.Sequential(
            discriminator_block(in_channels, 64),   # Input: [B, 2, 256, 256] -> Output: [B, 64, 128, 128]
            discriminator_block(64, 128),           # [B, 64, 128, 128] -> [B, 128, 64, 64]
            discriminator_block(128, 256),          # [B, 128, 64, 64] -> [B, 256, 32, 32]
            discriminator_block(256, 512, stride=1),# [B, 256, 32, 32] -> [B, 512, 30, 30]
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))  # [B, 512, 30, 30] -> [B, 1, 30, 30]
        )

    def forward(self, input_edges, edge):
        """
        Forward pass for the discriminator.
        
        Args:
        - input_edges: Masked edge map (1 channel)
        - edge: Ground truth edges (gt_edges) during training, predicted edges (pred_edge) during eval
        
        Returns:
        - Patch-based logits for real/fake classification.
        """
        x = torch.cat((input_edges, edge), dim=1)  # Concatenate along channel axis
        return self.model(x)  # Output shape: [B, 1, 30, 30]


def adversarial_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

def feature_matching_loss(disc_real, disc_fake):
    return torch.mean(torch.abs(disc_real - disc_fake))

def l1_loss(pred, target):
    return F.l1_loss(pred, target)