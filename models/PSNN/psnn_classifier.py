import torch.nn as nn
import torch
class Classifier(nn.Module):
    def __init__(self,num_classes,dimension, input_size=32768, hidden_size=128):
        super(Classifier, self).__init__()
        if dimension == 'DWT':
            self.classifier = nn.Sequential(
                nn.Linear(in_features=256*128,out_features=256),
                nn.Linear(in_features=256,out_features=128),
                nn.Linear(in_features=128,out_features=num_classes)
            )
        else: self.classifier = nn.Sequential(nn.Linear(in_features=1344,out_features=num_classes), nn.Dropout(0.5))

    

    def forward(self, x):
        # 计算第一层的均值和标准差
        outputs = self.classifier(x)

        return outputs