import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1
# Define the discriminator
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

class Discriminator(nn.Module):
    def __init__(self, dimension,num_out=1, max_iter=10000.0, trade_off_adversarial='Cons', lam_adversarial=1.0):
        super(Discriminator, self).__init__()
        if dimension == "DWT":
            self.domain_classifier = nn.Sequential(
                nn.Linear(32768, 128, nn.Dropout(0.5)),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, num_out)
                )
            
            self.sigmoid = nn.Sigmoid()
        else:
            self.domain_classifier = nn.Sequential(
                nn.Linear(1344, 128, nn.Dropout(0.5)),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, num_out)
                )
            
            self.sigmoid = nn.Sigmoid()

        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial

    def forward(self, x):
        
        x = x * 1.0
        x.register_hook(grl_hook(1))
        x = self.domain_classifier(x)
        x = self.sigmoid(x)
        return x
