import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self,num_class):
        super(Classifier, self).__init__()
        self.fc1_cls = nn.Linear(128,128)
        self.dropout1_cls = nn.Dropout(p=0.25)
        self.fc2_cls = nn.Linear(128,128)
        self.dropout2_cls = nn.Dropout(p=0.25)
        self.fc3_cls = nn.Linear(128, num_class)
        
    def forward(self, x):
        x = F.relu(self.fc1_cls(x))
        x = self.dropout1_cls(x)
        x = F.relu(self.fc2_cls(x))
        x = self.dropout2_cls(x)
        x = (self.fc3_cls(x))
        return (x)