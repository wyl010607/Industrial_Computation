import torch.nn as nn
import torch
class Classifier(nn.Module):
    def __init__(self,num_classes, input_size=1344, hidden_size=128):
        super(Classifier, self).__init__()
        self.fc1_mean = nn.Linear(input_size, hidden_size)
        self.fc1_std = nn.Linear(input_size, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, num_classes)
        self.fc2_std = nn.Linear(hidden_size, num_classes)

    def reparametrize(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + std * epsilon

    def forward(self, x):
        # 计算第一层的均值和标准差
        fc1_mean = self.fc1_mean(x)
        fc1_std = torch.exp(self.fc1_std(x))  # 使用对数标准差以确保为正

        # 为第一层进行重参数化
        fc1_out = self.reparametrize(fc1_mean, fc1_std)

        # 计算第二层的均值和标准差
        fc2_mean = self.fc2_mean(fc1_out)
        fc2_std = torch.exp(self.fc2_std(fc1_out))

        # 为第二层进行重参数化
        fc2_out = self.reparametrize(fc2_mean, fc2_std)

        return fc2_out