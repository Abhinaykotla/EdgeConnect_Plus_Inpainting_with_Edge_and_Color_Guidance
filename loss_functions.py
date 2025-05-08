import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models


def adversarial_loss(pred, target):
    """
    Computes binary cross entropy adversarial loss between predictions and targets.
    
    Used for training GANs to measure how well the discriminator distinguishes
    between real and generated samples.
    
    Args:
        pred (torch.Tensor): Discriminator predictions of shape [B, 1, H, W]
        target (torch.Tensor): Target labels of shape [B, 1, H, W]
                              (typically filled with 1s for real or 0s for fake)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    return F.binary_cross_entropy_with_logits(pred, target)


def feature_matching_loss(real_features, fake_features):
    """
    Compute feature matching loss between real and fake feature maps.
    
    This loss encourages the generator to produce images that match
    the feature statistics of real images at multiple levels of abstraction.
    
    Args:
        real_features (torch.Tensor or list): Feature maps from real images
        fake_features (torch.Tensor or list): Feature maps from generated images
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Handle single tensor case (for G1)
    if not isinstance(real_features, list) and not isinstance(fake_features, list):
        return torch.mean(torch.abs(real_features - fake_features))
    
    # Handle list of features case (for G2)
    # Sum L1 distances between each pair of features
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss


def l1_loss(pred, target):
    """
    Computes L1 loss between predictions and targets with normalization.
    
    Handles range inconsistencies between prediction and target by
    normalizing prediction from [-1,1] to [0,1] if needed.
    
    Args:
        pred (torch.Tensor): Predicted images, typically in range [-1,1]
        target (torch.Tensor): Target images, typically in range [0,1]
    
    Returns:
        torch.Tensor: Scalar L1 loss value
    """
    # If pred is in [-1,1] and target is in [0,1], normalize pred
    pred_normalized = (pred + 1) / 2
    return F.l1_loss(pred_normalized, target)


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 feature extractor for perceptual and style losses.
    
    Extracts intermediate feature maps from specific layers of a pre-trained VGG16 model.
    Used to compute perceptual and style losses between generated and ground truth images.
    
    Args:
        layers (tuple): Indices of VGG layers to extract features from (default: (3, 8, 15, 22))
                        These correspond to specific activation layers in VGG16
    """
    def __init__(self, layers=(3, 8, 15, 22)):
        super(VGG16FeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.selected_layers = layers
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layers)+1])
        # Freeze VGG parameters to use as a fixed feature extractor
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the VGG16 feature extractor.
        
        Args:
            x (torch.Tensor): Input images of shape [B, 3, H, W] in range [0,1]
        
        Returns:
            list: List of feature maps from selected VGG16 layers
        """
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features  # List of feature maps


def perceptual_loss(vgg, gen_img, gt_img):
    """
    Computes perceptual loss using VGG16 feature maps.
    
    Measures the L1 distance between VGG feature maps of generated and ground truth images.
    This loss encourages the network to generate images that are perceptually similar
    to the target images at various levels of abstraction.
    
    Args:
        vgg (VGG16FeatureExtractor): VGG16 feature extractor module
        gen_img (torch.Tensor): Generated images of shape [B, 3, H, W], range [-1,1]
        gt_img (torch.Tensor): Ground truth images of shape [B, 3, H, W], range [0,1]
    
    Returns:
        torch.Tensor: Scalar perceptual loss value
    """
    # Normalize predicted image from [-1,1] to [0,1]
    gen_img_normalized = (gen_img + 1) / 2
    
    # Extract VGG features for both images
    gen_features = vgg(gen_img_normalized)
    gt_features = vgg(gt_img)
    
    # Compute L1 distance between feature maps
    loss = 0.0
    for gf, gt in zip(gen_features, gt_features):
        loss += F.l1_loss(gf, gt)
    return loss


def gram_matrix(feat):
    """
    Compute Gram matrix from feature maps for style loss calculation.
    
    The Gram matrix captures style information by computing correlations
    between different feature channels.
    
    Args:
        feat (torch.Tensor): Feature tensor of shape [B, C, H, W]
    
    Returns:
        torch.Tensor: Gram matrix of shape [B, C, C], normalized by feature dimensions
    """
    (b, c, h, w) = feat.size()
    feat = feat.view(b, c, h * w)  # Reshape to [B, C, H*W]
    gram = torch.bmm(feat, feat.transpose(1, 2))  # Batch matrix multiplication: [B, C, C]
    # Add a small epsilon to avoid division by zero
    divisor = c * h * w + 1e-8
    return gram / divisor  # Normalize by feature dimensions


def style_loss(vgg, gen_img, gt_img):
    """
    Compute style loss between generated and ground truth images.
    
    Measures the L1 distance between Gram matrices of VGG feature maps.
    This loss encourages matching the texture and style statistics of the target.
    
    Args:
        vgg (VGG16FeatureExtractor): VGG16 feature extractor module
        gen_img (torch.Tensor): Generated images of shape [B, 3, H, W], range [-1,1]
        gt_img (torch.Tensor): Ground truth images of shape [B, 3, H, W], range [0,1]
    
    Returns:
        torch.Tensor: Scalar style loss value
    """
    # Normalize both images to same range before VGG processing
    # If your network outputs [-1,1] but ground truth is [0,1]
    gen_img_normalized = (gen_img + 1) / 2  # Convert from [-1,1] to [0,1]
    
    # Now both images are in [0,1] range
    gen_features = vgg(gen_img_normalized)
    gt_features = vgg(gt_img)

    loss = 0.0
    for gf, gt in zip(gen_features, gt_features):
        # Add safety check for valid feature maps (avoid NaN)
        if torch.isnan(gf).any() or torch.isnan(gt).any():
            continue
            
        # Compute Gram matrices
        gram_gf = gram_matrix(gf)
        gram_gt = gram_matrix(gt)
        
        # Skip this layer if NaN appears in gram matrices
        if torch.isnan(gram_gf).any() or torch.isnan(gram_gt).any():
            continue
            
        loss += F.l1_loss(gram_gf, gram_gt)
    
    # Make sure we don't return NaN even if all layers were skipped
    if torch.isnan(loss):
        return torch.tensor(0.0, device=loss.device)
        
    return loss


class InceptionV3Features:
    """
    Feature extractor based on InceptionV3 for FID calculation.
    
    Uses the standard approach for FID: the 2048-dimensional output of the
    final pooling layer before the classification layer.
    """
    def __init__(self, device):
        self.device = device
        # Load InceptionV3 pretrained model
        self.model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=True)
        # Remove classification layer
        self.model.fc = torch.nn.Identity()
        self.model.to(device)
        self.model.eval()
    
    def get_features(self, images):
        """
        Extract inception features from images.
        
        Args:
            images (torch.Tensor): Batch of images, shape [B, 3, H, W], range [0,1]
        
        Returns:
            torch.Tensor: Features of shape [B, 2048]
        """
        # Ensure images are in the right range [0, 1]
        if images.min() < 0:
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
        # Resize to InceptionV3 expected input size (299x299)
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            try:
                # Forward pass through the model
                features = self.model(images)
                return features
            except Exception as e:
                print(f"WARNING: Error extracting Inception features: {e}")
                return torch.zeros(images.shape[0], 2048, device=images.device)


def calculate_fid(real_features, fake_features):
    """
    Calculate Fréchet Inception Distance (FID) between two sets of features.
    
    FID measures the distance between the feature distributions of real and generated images,
    which correlates well with human perception of image quality.
    
    Args:
        real_features (torch.Tensor): Features from real images, shape [N, feat_dim]
        fake_features (torch.Tensor): Features from fake images, shape [M, feat_dim]
    
    Returns:
        float: FID score (lower is better)
    """
    # Convert to numpy
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()
    
    # Ensure features are properly shaped for covariance calculation
    if real_features.ndim > 2:
        real_features = real_features.reshape(real_features.shape[0], -1)
    if fake_features.ndim > 2:
        fake_features = fake_features.reshape(fake_features.shape[0], -1)
    
    # Calculate mean and covariance statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute squared distance between means
    mean_diff_squared = np.sum((mu_real - mu_fake) ** 2)
    
    # Add small epsilon to diagonal for numerical stability
    epsilon = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * epsilon
    sigma_fake += np.eye(sigma_fake.shape[0]) * epsilon
    
    # Compute matrix sqrt with SVD which is more stable
    sigma_real_sqrt = linalg.sqrtm(sigma_real)
    product = np.matmul(sigma_real_sqrt, sigma_fake)
    product = np.matmul(sigma_real_sqrt, product)
    
    # Numerical stability check for complex numbers
    if np.iscomplexobj(product):
        product = product.real
    
    # Calculate trace term
    trace_term = np.trace(sigma_real + sigma_fake - 2 * linalg.sqrtm(product))
    
    # FID formula
    fid = mean_diff_squared + trace_term
    
    return float(fid)