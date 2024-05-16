import torch.nn as nn
from .Backbone import ResNet1D, MLPNet, CNN1D
import torchvision.models as models
class Feature_extractor(nn.Module):
    def __init__(self, backbone,dimension):
        super(Feature_extractor, self).__init__()
        if dimension == '1D':
            if backbone == "ResNet1D":
                self.encoder = ResNet1D.resnet18()
            elif backbone == "ResNet2D":
                self.model_ft = models.resnet18(pretrained=True)
                self.bottleneck = nn.Sequential(nn.Linear(self.model_ft.fc.out_features, 512), nn.ReLU(), nn.Dropout(0.5))
                self.encoder = nn.Sequential(self.model_ft, self.bottleneck)
            elif backbone == "MLPNet":
                self.encoder = MLPNet.MLPNet()
            elif backbone == "CNN1D":
                self.encoder = CNN1D.CNN1D()
            else:
                raise Exception("model not implement")
        elif dimension == 'DWT':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
                nn.AdaptiveMaxPool2d((16,32)),
                nn.Flatten() 
            )

    def forward(self, x):
        
        logits = self.encoder(x)

        return logits