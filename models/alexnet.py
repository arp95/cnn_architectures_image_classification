# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
from convolutions import *


# AlexNet CNN Architecture (remember input size: (224 x 224 x 3))
class AlexNet(torch.nn.Module):
    
    # init function
    def __init__(self, num_classes = 2):
        super(AlexNet, self).__init__()
        
        # first part - find conv-features
        self.features = torch.nn.Sequential(
   
            # first conv block
            Convolution(3, 64, 11, 4, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # second conv block
            Convolution(64, 192, 5, 1, 2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # third conv block
            Convolution(192, 384, 3, 1, 1),
            Convolution(384, 256, 3, 1, 1),
            Convolution(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # second part - apply adaptive average pool before flattening the conv-features into one vector
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        
        # third part - flatten the conv-features and apply a series of fc layers followed by the output layer
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes)
        )
        

    # forward propagation step
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
