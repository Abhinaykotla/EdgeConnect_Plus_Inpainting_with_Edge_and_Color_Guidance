import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
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
        return x + self.block(x)

class InpaintingGeneratorG2(nn.Module):
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

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(base_channels * 4) for _ in range(num_res_blocks)])

        # Upsample layers
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

        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=3)
        self.activation = nn.Tanh()  # Outputs in range [-1, 1]

    def forward(self, input_img, guidance_img, mask):
        x = torch.cat((input_img, guidance_img, mask), dim=1)  # (B, 7, H, W)
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.activation(self.final_conv(x))


class InpaintDiscriminatorG2(nn.Module):
    def __init__(self, in_channels=6):
        super(InpaintDiscriminatorG2, self).__init__()

        def disc_block(in_c, out_c, stride=2):
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

    def forward(self, input_img, gen_or_gt_img):
        x = torch.cat((input_img, gen_or_gt_img), dim=1)  # (B, 6, H, W)
        return self.model(x)
