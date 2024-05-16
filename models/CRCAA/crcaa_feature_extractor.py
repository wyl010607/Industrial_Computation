import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
class Feature_extractor(nn.Module):
    def __init__(self,dimension):
        super(Feature_extractor, self).__init__()
        if dimension == 'DWT':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2),
                nn.AdaptiveMaxPool2d((16,32)),
                nn.Flatten(),
                nn.Linear(128*512,512),
                nn.ReLU(),
                nn.Linear(512,128),
            )
        else:
            self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=4,padding=0),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=3, stride=2,padding=1),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),


            nn.MaxPool1d(kernel_size=3, stride=2,padding=1),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, stride=1,padding=2),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Flatten(),
            )
        


    def forward(self, x):
        
        # Flatten
        feature = self.encoder(x)

        
        
        
        return feature