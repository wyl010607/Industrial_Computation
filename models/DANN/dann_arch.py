from math import ceil

import torch
import torch.nn as nn

from .dann_feature_extractor import Feature_extractor
from .dann_classifier import Classifier
from .dann_discriminator import Discriminator


class DANN(nn.Module):
    def __init__(self, **model_args):
        super(DANN, self).__init__()

        self.backbone = str(model_args["backbone"])
        self.class_num = model_args["class_num"]
        self.dimension = model_args["dimension"]

        # Encoder
        self.feature_extractor = Feature_extractor(self.backbone,self.dimension)

        # Classifier
        self.classifier = Classifier(self.backbone, self.dimension,self.class_num)

        # Domain Discriminator
        self.domain_discriminator = Discriminator(self.backbone,self.dimension)

    def forward(self, data: torch.Tensor,train: bool, **kwargs):
        if train:
            features = self.feature_extractor(data)
            class_logits = self.classifier(features)
            domain_labels = self.domain_discriminator(features)
            return domain_labels, class_logits
        else:
            features = self.feature_extractor(data)
            class_logits = self.classifier(features)
            return class_logits

