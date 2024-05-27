from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self,n_features):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_features, 100)
        self.linear2 = nn.Linear(100,100)
        self.output = nn.Linear(100,1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        out = self.relu(self.output(x))
        return out