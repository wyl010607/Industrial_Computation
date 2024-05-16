import torch.nn as nn

import torchvision.models as models
class Feature_extractor(nn.Module):
    def __init__(self,dimension):
        super(Feature_extractor, self).__init__()
        if dimension == '1D':
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(in_channels=16, out_channels=32,kernel_size=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(output_size=4),
            )
            self.high_low = nn.Sequential(
                nn.Linear(128 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            )
            
        elif dimension == 'DWT':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3),
                nn.ReLU,
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.ReLU,
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
                nn.ReLU,
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.ReLU,
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
                nn.ReLU,
                nn.AdaptiveMaxPool2d((16,32)),
                nn.ReLU,
                nn.Flatten(),
                nn.Linear(128 * 256, 512),
                nn.ReLU()
            )
             
            self.high_low = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
            )


    def forward(self, x):
        
        
        high_feature = self.encoder(x)

        low_feature = self.high_low(x)
        return low_feature,high_feature