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
        super(net, self).__init__()
        res152 = models.resnet152(pretrained=True)
        modules = list(res152.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(res152.fc.in_features, 43)
        self.bn = nn.BatchNorm1d(43, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images.cuda())
        features = features.reshape(features.size(0), -1).cuda()
        features = self.bn(self.linear(features))
        features.type(torch.FloatTensor).cuda()
        return features


