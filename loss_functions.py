import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
