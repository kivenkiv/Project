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

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 6, 5)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(6, 16, 5)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.pool2(self.relu4(self.conv4(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)
        return x
  
