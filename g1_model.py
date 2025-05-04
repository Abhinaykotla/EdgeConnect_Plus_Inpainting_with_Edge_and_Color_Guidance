# g1_model.py - Defines the first generator (G1) and discriminator (D1) models
# G1 generates edge maps from masked inputs, while D1 evaluates the authenticity of those edges

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A residual block with dilated convolutions for expanded receptive field.
    
    Dilated convolutions allow the network to capture wider spatial context
    while maintaining computational efficiency.
    
    Args:
        in_channels (int): Number of input channels
        
    Returns:
        torch.Tensor: Output tensor with same shape as input after residual connection
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, in_channels, H, W]
        """
        residual = x  # Store input for residual connection
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual  # Skip connection adds original features to processed features

class EdgeGenerator(nn.Module):
    """
    Edge Generator (G1) for EdgeConnect architecture.
    
    Uses an encoder-decoder structure with residual blocks in the middle.
    Takes masked images as input and predicts complete edge maps.
    
    Network Structure:
    1. Initial convolution to extract features
    2. Downsampling to reduce spatial dimensions
    3. Residual blocks for feature processing
    4. Upsampling to restore original resolution
    5. Final convolution with sigmoid to output normalized edge probabilities
    """
    def __init__(self):
        super(EdgeGenerator, self).__init__()

        # Initial Convolution - Expands 3 input channels to 64 feature channels
        # Input: Masked RGB image and mask concatenated (3 channels total)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling - Reduces spatial dimensions while increasing features
        # First downsample: 64 -> 128 channels, H/2 x W/2 resolution
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Second downsample: 128 -> 256 channels, H/4 x W/4 resolution
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual Blocks - Process features at reduced resolution
        # Maintains channel count (256) but increases effective receptive field
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])

        # Upsampling - Increases spatial dimensions while reducing features
        # First upsample: 256 -> 128 channels, H/2 x W/2 resolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Second upsample: 128 -> 64 channels, H x W resolution (original)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final Output - Converts feature maps to single-channel edge probability map
        self.final_conv = nn.Conv2d(64, 1, kernel_size=7, padding=3)
        self.activation = nn.Sigmoid()  # Normalizes outputs to [0,1] range for edge probabilities

    def forward(self, img, mask, gray):
        """
        Forward pass of the edge generator.
        
        Args:
            img (torch.Tensor): Input image tensor of shape [B, 1, H, W]
            mask (torch.Tensor): Mask tensor of shape [B, 1, H, W], where 1=valid, 0=hole
            gray (torch.Tensor): Grayscale image tensor of shape [B, 1, H, W]
            
        Returns:
            torch.Tensor: Predicted edge map of shape [B, 1, H, W] with values in [0,1]
        """
        x = torch.cat((img, mask, gray), dim=1)  # Combine inputs to create 3-channel input
        x = self.init_conv(x)                   # [B, 3, H, W] -> [B, 64, H, W]
        x = self.down1(x)                       # [B, 64, H, W] -> [B, 128, H/2, W/2]
        x = self.down2(x)                       # [B, 128, H/2, W/2] -> [B, 256, H/4, W/4]
        x = self.res_blocks(x)                  # [B, 256, H/4, W/4] -> [B, 256, H/4, W/4]
        x = self.up1(x)                         # [B, 256, H/4, W/4] -> [B, 128, H/2, W/2]
        x = self.up2(x)                         # [B, 128, H/2, W/2] -> [B, 64, H, W]
        x = self.final_conv(x)                  # [B, 64, H, W] -> [B, 1, H, W]
        return self.activation(x)               # Apply sigmoid to get probability map


class EdgeDiscriminator(nn.Module):
    """
    PatchGAN-based Discriminator (D1) for Edge Generation.
    
    Classifies if edge maps are real or generated by examining patches of the input.
    Uses spectral normalization for training stability.
    
    Args:
        in_channels (int): Number of input channels (default: 3)
            Typically combines masked edges, grayscale image, and target/predicted edges
    """
    def __init__(self, in_channels=3):
        super(EdgeDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2):
            """
            Creates a basic discriminator block with spectral normalization.
            
            Args:
                in_filters (int): Number of input channels
                out_filters (int): Number of output channels
                stride (int): Stride value for the convolution (default: 2)
                
            Returns:
                nn.Sequential: A block with spectral normalized conv and LeakyReLU
            """
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # Progressive downsampling with spectral normalization for stability
        self.model = nn.Sequential(
            discriminator_block(in_channels, 64),   # Input: [B, 3, 256, 256] -> Output: [B, 64, 128, 128]
            discriminator_block(64, 128),           # [B, 64, 128, 128] -> [B, 128, 64, 64]
            discriminator_block(128, 256),          # [B, 128, 64, 64] -> [B, 256, 32, 32]
            discriminator_block(256, 512, stride=1),# [B, 256, 32, 32] -> [B, 512, 30, 30]
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))  # [B, 512, 30, 30] -> [B, 1, 30, 30]
        )

    def forward(self, input_edges, gray, edge):
        """
        Forward pass for the discriminator.
        
        Args:
            input_edges (torch.Tensor): Masked/incomplete edge map of shape [B, 1, H, W]
            gray (torch.Tensor): Grayscale image tensor of shape [B, 1, H, W]
            edge (torch.Tensor): Ground truth or predicted edge map of shape [B, 1, H, W]
        
        Returns:
            torch.Tensor: Patch-based logits for real/fake classification of shape [B, 1, 30, 30]
                        Each value represents the authenticity of a corresponding image patch
        """
        x = torch.cat((input_edges, gray, edge), dim=1)  # Combine inputs into 3-channel tensor
        return self.model(x)  # Process through discriminator network