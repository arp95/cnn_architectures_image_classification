# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2


# define ResNet_50 network (remember input size: (224 x 224 x 3))
class ResNet_50(torch.nn.Module):
    
    
    # block 1
    def block_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256)
        )    
        
    # block 2
    def block_2_init(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 128, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512)
        )
    
    def block_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512)
        )
    
    # block 3
    def block_3_init(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 256, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 1024, kernel_size=1),
            torch.nn.BatchNorm2d(1024)
        )
    
    def block_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 1024, kernel_size=1),
            torch.nn.BatchNorm2d(1024)
        )
    
    # block 4
    def block_4_init(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 512, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 2048, kernel_size=1),
            torch.nn.BatchNorm2d(2048)
        )
    
    def block_4(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 2048, kernel_size=1),
            torch.nn.BatchNorm2d(2048)
        )
    
    
    # init function
    def __init__(self, num_classes = 2):
        super(ResNet_50, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # block 1
        self.resnet_block_1_1 = self.block_1(64)
        self.resnet_block_1_2 = self.block_1(256)
        self.resnet_block_1_3 = self.block_1(256)
        
        # block 2
        self.resnet_block_2_1 = self.block_2_init(256)
        self.resnet_block_2_2 = self.block_2(512)
        self.resnet_block_2_3 = self.block_2(512)
        self.resnet_block_2_4 = self.block_2(512)
        
        # block 3
        self.resnet_block_3_1 = self.block_3_init(512)
        self.resnet_block_3_2 = self.block_3(1024)
        self.resnet_block_3_3 = self.block_3(1024)
        self.resnet_block_3_4 = self.block_3(1024)
        self.resnet_block_3_5 = self.block_3(1024)
        self.resnet_block_3_6 = self.block_3(1024)
        
        # block 4
        self.resnet_block_4_1 = self.block_4_init(1024)
        self.resnet_block_4_2 = self.block_4(2048)
        self.resnet_block_4_3 = self.block_4(2048)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d(7)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048 * 7 * 7, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
        
        self.relu = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True)
        )
        
        self.skip_connection_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256)
        )
        
        self.skip_connection_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(512)
        )
        
        self.skip_connection_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(1024)
        )
        
        self.skip_connection_4 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 2048, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(2048)
        )
        
        
    # define forward function
    def forward(self, x):
        
        # apply initial conv layers
        x = self.features(x)
        
        # block 1
        x_1 = self.resnet_block_1_1(x)
        x = self.skip_connection_1(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_1_2(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_1_3(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        
        # block 2
        x_1 = self.resnet_block_2_1(x)
        x = self.skip_connection_2(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_2_2(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_2_3(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_2_4(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        
        # block 3
        x_1 = self.resnet_block_3_1(x)
        x = self.skip_connection_3(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_3_2(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_3_3(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_3_4(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_3_5(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_3_6(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        
        # block 4
        x_1 = self.resnet_block_4_1(x)
        x = self.skip_connection_4(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_4_2(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        x_1 = self.resnet_block_4_3(x)
        x = torch.add(x, x_1)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
