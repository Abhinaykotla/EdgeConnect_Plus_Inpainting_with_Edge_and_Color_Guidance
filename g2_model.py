import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block for the G2 inpainting generator.
    
    Implements a standard residual block with two convolutional layers and skip connection,
    helping mitigate the vanishing gradient problem in deep networks.
    
    Args:
        in_channels (int): Number of input channels
        
    Returns:
        torch.Tensor: Output tensor with same dimensions as input after adding residual connection
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        """
        Forward pass through residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, in_channels, H, W]
        """
        return x + self.block(x)  # Skip connection adds input to processed features

class InpaintingGeneratorG2(nn.Module):
    """
    Generator (G2) for the second stage of EdgeConnect+ architecture.
    
    Takes masked RGB images, guidance (edge and color maps), and masks as input,
    and produces inpainted RGB images as output.
    
    Architecture follows an encoder-decoder structure with residual blocks:
    1. Encoder: Initial convolution followed by downsampling
    2. Transformation: Multiple residual blocks for feature processing
    3. Decoder: Upsampling to restore original resolution
    
    Args:
        in_channels (int): Number of input channels (default: 7 - 3 for RGB image, 3 for guidance, 1 for mask)
        out_channels (int): Number of output channels (default: 3 for RGB image)
        base_channels (int): Number of base channels (default: 64)
        num_res_blocks (int): Number of residual blocks (default: 8)
    """
    def __init__(self, in_channels=7, out_channels=3, base_channels=64, num_res_blocks=8):
        super(InpaintingGeneratorG2, self).__init__()

        # Initial conv layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Downsample layers
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # Residual blocks - processes features at reduced resolution
        # Maintains channel count but expands effective receptive field
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels * 4) for _ in range(num_res_blocks)])

        # Upsample layers - restore spatial resolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Final output layer - project to RGB channels
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=3)
        self.activation = nn.Tanh()  # Outputs in range [-1, 1]

    def forward(self, input_img, guidance_img, mask):
        """
        Forward pass of the inpainting generator.
        
        Args:
            input_img (torch.Tensor): Input masked image tensor of shape [B, 3, H, W]
            guidance_img (torch.Tensor): Guidance image tensor (edges and colors) of shape [B, 3, H, W]
            mask (torch.Tensor): Binary mask tensor of shape [B, 1, H, W] where 1=valid, 0=hole
            
        Returns:
            torch.Tensor: Inpainted RGB image of shape [B, 3, H, W] with values in [-1, 1]
        """
        x = torch.cat((input_img, guidance_img, mask), dim=1)  # (B, 7, H, W)
        x = self.init_conv(x)  # [B, 7, H, W] -> [B, 64, H, W]
        x = self.down1(x)     # [B, 64, H, W] -> [B, 128, H/2, W/2]
        x = self.down2(x)     # [B, 128, H/2, W/2] -> [B, 256, H/4, W/4]
        x = self.res_blocks(x) # [B, 256, H/4, W/4] -> [B, 256, H/4, W/4]
        x = self.up1(x)       # [B, 256, H/4, W/4] -> [B, 128, H/2, W/2]
        x = self.up2(x)       # [B, 128, H/2, W/2] -> [B, 64, H, W]
        return self.activation(self.final_conv(x))  # [B, 64, H, W] -> [B, 3, H, W]


class InpaintDiscriminatorG2(nn.Module):
    """
    Discriminator for the G2 inpainting generator.
    
    Uses a PatchGAN architecture with spectral normalization for training stability.
    Evaluates the realism of inpainted regions by classifying if generated patches are real or fake.
    
    Args:
        in_channels (int): Number of input channels (default: 6 = 3 for input + 3 for ground truth/generated)
    """
    def __init__(self, in_channels=6):
        super(InpaintDiscriminatorG2, self).__init__()

        def disc_block(in_c, out_c, stride=2):
            """
            Helper function to create a discriminator block with spectral normalization.
            
            Args:
                in_c (int): Number of input channels
                out_c (int): Number of output channels
                stride (int): Stride value for downsampling (default: 2)
                
            Returns:
                nn.Sequential: A block with spectral normalized conv and LeakyReLU
            """
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            disc_block(in_channels, 64),      # → [B, 64, 128, 128]
            disc_block(64, 128),              # → [B, 128, 64, 64]
            disc_block(128, 256),             # → [B, 256, 32, 32]
            disc_block(256, 512, stride=1),   # → [B, 512, 31, 31]
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1))  # → [B, 1, 30, 30]
        )

    def get_features(self, input_img, image):
        """
        Extract intermediate features for feature matching loss.
        
        Args:
            input_img (torch.Tensor): Input masked image tensor of shape [B, 3, H, W]
            image (torch.Tensor): Generated or ground truth image of shape [B, 3, H, W]
            
        Returns:
            list: List of feature tensors from intermediate layers
        """
        # Concatenate input image and target/generated image
        x = torch.cat([input_img, image], dim=1)
        features = []
        
        # Store intermediate layer outputs
        for i, layer in enumerate(self.model):
            x = layer(x)
            # Store features from some layers (adjust as needed)
            if i % 2 == 0 and i > 0:  # Store every other layer after the first
                features.append(x)
                
        return features

    def forward(self, input_img, gen_or_gt_img, return_features=False):
        """
        Forward pass through the discriminator.
        
        Args:
            input_img (torch.Tensor): Input masked image tensor of shape [B, 3, H, W]
            gen_or_gt_img (torch.Tensor): Generated or ground truth image of shape [B, 3, H, W]
            return_features (bool): Whether to return intermediate features (default: False)
            
        Returns:
            torch.Tensor or tuple: 
                - If return_features=False: Final discriminator outputs of shape [B, 1, 30, 30]
                - If return_features=True: Tuple of (outputs, features list)
        """
        x = torch.cat((input_img, gen_or_gt_img), dim=1)  # (B, 6, H, W)
        features = []

        # Pass through each layer and collect intermediate features
        for module in self.model:
            x = module(x)
            if return_features:
                features.append(x)

        if return_features:
            return x, features  # Return final output and intermediate features
        else:
            return x  # Return only the final output