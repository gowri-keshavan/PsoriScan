"""ResNet18 transfer-learning model in PyTorch."""
import torch.nn as nn
from torchvision import models


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """ResNet18 with a single-logit head. Apply sigmoid to get P(psoriasis)."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    return model
