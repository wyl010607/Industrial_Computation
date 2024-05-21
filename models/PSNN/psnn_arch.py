from math import ceil

import torch
import torch.nn as nn

from .psnn_feature_extractor import Feature_extractor
from .psnn_classifier import Classifier
from .psnn_discriminator import Discriminator


class PSNN(nn.Module):
    def __init__(self, **model_args):
        super(PSNN, self).__init__()

        self.class_num = model_args["class_num"]
        self.dimension = model_args["dimension"]

        # Encoder
        self.feature_extractor = Feature_extractor(self.dimension)
        

        # Classifier
        self.classifier1 = Classifier(self.class_num,self.dimension)
        self.classifier2 = Classifier(self.class_num,self.dimension)

        # Domain Discriminator
        self.domain_discriminator = Discriminator(self.dimension)

    def forward(self, data: torch.Tensor,train: bool, **kwargs):
        if train:
            features = self.feature_extractor(data)
            class_logits1 = self.classifier1(features)
            class_logits2 = self.classifier2(features)
            domain_labels = self.domain_discriminator(features)
            return domain_labels, class_logits1,class_logits2
        else:
            features = self.feature_extractor(data)
            class_logits = self.classifier1(features)
            return class_logits

