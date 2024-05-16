import torch 
import torch.nn as nn

from .dpada_feature_extractor import Feature_extractor
from .dpada_classifier import Classifier
from .dpada_discriminator import Discriminator

class DPADA(nn.Module):
    def __init__(self, **model_args):
        super(DPADA, self).__init__()

        self.class_num = model_args["class_num"]
        self.dimension = model_args["dimension"]

        # Encoder
        self.feature_extractor = Feature_extractor(self.dimension)

        # Classifier
        self.classifier = Classifier(self.class_num)

        # Domain Discriminator
        self.domain_discriminator = Discriminator(self.class_num)

    def forward(self, data: torch.Tensor, **kwargs):
        
            low_features,high_features = self.feature_extractor(data)
            class_logits = self.classifier(low_features)
            domain_labels = self.domain_discriminator(low_features,class_logits)
            return domain_labels, class_logits,high_features
        
