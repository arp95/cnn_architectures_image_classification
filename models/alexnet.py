# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np


# Define AlexNet CNN Architecture (remember input size: (224 x 224 x 3))
class AlexNet(torch.nn.Module):
    
    # init function
    def __init__(self, num_classes = 2):
        super(AlexNet, self).__init__()
        
        # first part - find conv-features
        self.features = torch.nn.Sequential(
   
            # first conv block
            torch.nn.Conv2d(3, 64, kernel_size=11, padding=2, stride=4),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            # second conv block
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            # third conv block
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # second part - apply adaptive average pool before flattening the conv-features into one vector
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        
        # third part - flatten the conv-features and apply a series of fc layers followed by the output layer
        self.classifier = torch.nn.Sequential(

            # first fc layer
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),

            # second fc layer
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),

            # third fc layer
            torch.nn.Linear(4096, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
        

    # forward propagation step
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
