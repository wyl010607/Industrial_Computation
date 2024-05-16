import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def grl_hook(coeff=1.0):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class Discriminator(nn.Module):
    def __init__(self, num_class,input_size_M=256):
        super(Discriminator, self).__init__()
        self.fc1_disc = nn.Linear(128, 128)
        self.dropout1_disc = nn.Dropout(p=0.25)
        self.fc2_disc = nn.Linear(128, 128)
        self.dropout2_disc = nn.Dropout(p=0.25)
        self.fc3_disc = nn.Linear(128, 1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = x * 1.0
        x.register_hook(grl_hook())

        x = F.relu(self.fc1_disc(x))
        x = self.dropout1_disc(x)
        x = F.relu(self.fc2_disc(x))
        x = self.dropout2_disc(x)
        x = self.fc3_disc(x)
        return self.sigmoid(x)

