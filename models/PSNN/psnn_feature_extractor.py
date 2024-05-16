import torch.nn as nn

import torchvision.models as models
class Feature_extractor(nn.Module):
    def __init__(self,dimension):
        super(Feature_extractor, self).__init__()
        if dimension == 'DWT':
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
                nn.AdaptiveMaxPool2d((16,32)),
                nn.Flatten(), 
                nn.Linear(512*128,1344)
                
            )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(in_channels=1,out_channels=16, kernel_size=(32), stride=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(16, 32, kernel_size=(16), stride=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(32, 64, kernel_size=(5), stride=2),
                nn.MaxPool1d(kernel_size=3, stride=2),
                nn.Conv1d(64, 64, kernel_size=(5), stride=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Flatten()
            )


    def forward(self, x):
        feature = self.feature_extractor(x)

        return feature