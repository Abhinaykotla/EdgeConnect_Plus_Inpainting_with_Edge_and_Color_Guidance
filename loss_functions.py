import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models


def adversarial_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)

def feature_matching_loss(real_features, fake_features):
    """
    Compute feature matching loss between real and fake feature maps
    
    Args:
        real_features: Either a single tensor or list of feature maps
        fake_features: Either a single tensor or list of feature maps
    """
    # Handle single tensor case (for G1)
    if not isinstance(real_features, list) and not isinstance(fake_features, list):
        return torch.mean(torch.abs(real_features - fake_features))
    
    # Handle list of features case (for G2)
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss

def l1_loss(pred, target):
    # If pred is in [-1,1] and target is in [0,1], normalize pred
    pred_normalized = (pred + 1) / 2
    return F.l1_loss(pred_normalized, target)

# Load VGG16 Feature Extractor (Frozen)
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, layers=(3, 8, 15, 22)):
        super(VGG16FeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.selected_layers = layers
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:max(layers)+1])
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features  # List of feature maps

# Perceptual Loss L_perc
def perceptual_loss(vgg, gen_img, gt_img):
    # Normalize predicted image from [-1,1] to [0,1]
    gen_img_normalized = (gen_img + 1) / 2
    
    gen_features = vgg(gen_img_normalized)
    gt_features = vgg(gt_img)
    
    loss = 0.0
    for gf, gt in zip(gen_features, gt_features):
        loss += F.l1_loss(gf, gt)
    return loss

# Style Loss L_style
def gram_matrix(feat):
    (b, c, h, w) = feat.size()
    feat = feat.view(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))  # (B, C, C)
    # Add a small epsilon to avoid division by zero
    divisor = c * h * w + 1e-8
    return gram / divisor

def style_loss(vgg, gen_img, gt_img):
    # Normalize both images to same range before VGG processing
    # If your network outputs [-1,1] but ground truth is [0,1]
    gen_img_normalized = (gen_img + 1) / 2  # Convert from [-1,1] to [0,1]
    
    # Now both images are in [0,1] range
    gen_features = vgg(gen_img_normalized)
    gt_features = vgg(gt_img)

    loss = 0.0
    for gf, gt in zip(gen_features, gt_features):
        # Add safety check for valid gram matrices
        if torch.isnan(gf).any() or torch.isnan(gt).any():
            continue
            
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
    def __init__(self, device):
        self.device = device
        # Load pretrained Inception model
        self.inception_model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        self.inception_model.fc = torch.nn.Identity()  # Remove classification layer
        self.inception_model.to(device)
        self.inception_model.eval()
        
        # Register hook to get features
        self.features = None
        def hook(module, input, output):
            self.features = output.detach()
        
        # Register the hook on the mixed_7c layer (2048-dim features)
        # This is a more reliable place to extract features than avgpool
        self.inception_model.Mixed_7c.register_forward_hook(hook)
    
    def get_features(self, images):
        """Extract features from images"""
        # Resize images to Inception input size
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Ensure values are in [0, 1] and convert to [-1, 1] for Inception
        if images.min() < 0:
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        images = images * 2 - 1  # Convert to [-1, 1] for Inception
        
        # Get features
        with torch.no_grad():
            try:
                _ = self.inception_model(images)
            except Exception as e:
                print(f"Error getting Inception features: {e}")
                return torch.zeros(images.shape[0], 2048).to(images.device)
        
        # Reshape to 2D [batch_size, features]
        if self.features is not None:
            # Adaptive pooling to get a fixed-size feature vector
            features = self.features
            features = adaptive_avg_pool2d(features, (1, 1))
            features = features.reshape(features.shape[0], -1)
            return features
        else:
            print("Warning: No features captured from hook")
            return torch.zeros(images.shape[0], 2048).to(images.device)


def calculate_fid(real_features, fake_features):
    """Calculate FID between real and fake feature distributions"""
    # Convert to numpy and ensure features are properly shaped
    real_features = real_features.cpu().numpy()
    fake_features = fake_features.cpu().numpy()
    
    # Reshape if needed - we need 2D arrays for covariance calculation
    # Typically, Inception features are [batch_size, num_features, 1, 1] or similar
    if real_features.ndim > 2:
        real_features = real_features.reshape(real_features.shape[0], -1)
        fake_features = fake_features.reshape(fake_features.shape[0], -1)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    mu_diff = mu_real - mu_fake
    
    # Calculate sqrt of product of covariances
    # Add a small epsilon to the diagonal for numerical stability
    eps = 1e-6
    sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * eps
    sigma_fake = sigma_fake + np.eye(sigma_fake.shape[0]) * eps
    
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    
    # Check and correct imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = mu_diff @ mu_diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return fid
