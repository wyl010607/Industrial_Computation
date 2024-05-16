import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self, backbone, dimension,num_out=10):
        super(Classifier, self).__init__()
        self.backbone = backbone
        if dimension == 'DWT':
            self.classifier = nn.Sequential(
                nn.Linear(in_features=256*128,out_features=256),
                nn.Linear(in_features=256,out_features=128),
                nn.Linear(in_features=128,out_features=num_out)
            )
            
        else:
            if self.backbone in ("ResNet1D", "ResNet2D"):
                self.classifier = nn.Sequential(nn.Linear(in_features=512,out_features=num_out), nn.Dropout(0.5))
            if self.backbone in ("MLPNet", "CNN1D"):
                self.classifier = nn.Sequential(nn.Linear(in_features=64,out_features=num_out), nn.Dropout(0.5))
            




    def forward(self, logits):
        outputs = self.classifier(logits)

        return outputs