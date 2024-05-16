import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def grl_hook(coeff=1):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1
# Define the discriminator
def calc_coeff(iter_num=1, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

class Discriminator(nn.Module):
    def __init__(self, backbone,dimension,num_out=1, max_iter=10000.0, trade_off_adversarial='Cons', lam_adversarial=1.0):
        super(Discriminator, self).__init__()
        if dimension == "1D":
            if backbone in ("ResNet1D", "ResNet2D"):
                self.domain_classifier = nn.Sequential(
                    nn.Linear(512, 128, nn.Dropout(0.5)),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, num_out)
                )
            elif backbone in ("MLPNet", "CNN1D"):
                self.domain_classifier = nn.Sequential(
                    nn.Linear(64, 32, nn.Dropout(0.5)),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32, num_out)
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
        elif dimension == "DWT":
            self.domain_classifier = nn.Sequential(
                nn.Linear(in_features=256*128,out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256,out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128,out_features=1)
            )
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x = x * 1.0
        x.register_hook(grl_hook())
        x = self.domain_classifier(x)
        x = self.sigmoid(x)
        return x
