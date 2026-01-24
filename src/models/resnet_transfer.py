import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetTransfer(nn.Module):
    """ResNet18 transfer learning for CIFAR-10/Fashion-MNIST. Based on: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"""
    def __init__(self, num_classes=10, dropout=0.5, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        
        # Adapt for 32x32: adjust first conv and maxpool
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # For Fashion-MNIST (1 channel), add a conv to expand to 3
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool + fc
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
        
        # Xavier init for new fc layer
        self.fc.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        if x.size(1) == 1:  
            x = self.channel_adapter(x)
        x = self.features(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
