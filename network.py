import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd as autograd
from torch.autograd import Variable
import math

class net(nn.Module):
    ####
    # define your model
    ####
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(net, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
            print(features.shape)
        return features


