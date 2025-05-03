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
    return F.l1_loss(pred, target)

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
    gen_features = vgg(gen_img)
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
    return gram / (c * h * w)

def style_loss(vgg, gen_img, gt_img):
    gen_features = vgg(gen_img)
    gt_features = vgg(gt_img)

    loss = 0.0
    for gf, gt in zip(gen_features, gt_features):
        gram_gf = gram_matrix(gf)
        gram_gt = gram_matrix(gt)
        loss += F.l1_loss(gram_gf, gram_gt)
    return loss
