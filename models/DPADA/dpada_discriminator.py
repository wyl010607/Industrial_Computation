import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, num_class,input_size_M=256):
        super(Discriminator, self).__init__()
        # 第一层全连接层
        self.fc1 = nn.Linear(input_size_M + num_class, 256)
        # 第二层全连接层
        self.fc2 = nn.Linear(256, 1)  # 输出一个数值作为判别器输出

    def forward(self, low_feature, classify_result):
        # 将 M 和 C 的特征进行张量积，然后展平为一维向量
        
        x = torch.cat((classify_result, low_feature),dim=1)
        
        
        
        # 第一层全连接层
        x = F.relu(self.fc1(x))
        # 第二层全连接层，输出判别器认为来源于目标域的可能性
        domain_result = torch.sigmoid(self.fc2(x))
        return domain_result

