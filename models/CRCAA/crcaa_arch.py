import torch 
import torch.nn as nn

from .crcaa_feature_extractor import Feature_extractor
from .crcaa_classifier import Classifier
from .crcaa_discriminator import Discriminator

class CRCAA(nn.Module):
    def __init__(self, **model_args):
        super(CRCAA, self).__init__()

        self.class_num = model_args["class_num"]
        self.dimension = model_args["dimension"]

        # Encoder
        self.feature_extractor = Feature_extractor(self.dimension)

        # Classifier
        self.classifier = Classifier(self.class_num)

        # Domain Discriminator
        self.domain_discriminator = Discriminator(self.class_num)

    def forward(self, data: torch.Tensor,train: bool, **kwargs):
        if train:
            
            features = self.feature_extractor(data)
            
            class_logits = self.classifier(features)
            domain_labels = self.domain_discriminator(features)
            return domain_labels, class_logits,features
        else:
            features = self.feature_extractor(data)
            class_logits = self.classifier(features)
            return class_logits
