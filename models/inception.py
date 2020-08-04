# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2


# define InceptionNet network (remember input size: (224 x 224 x 3))
class InceptionNet(torch.nn.Module):
    
    # init function
    def __init__(self, num_classes = 2):
        super(InceptionNet, self).__init__()
        
        self.features = torch.nn.Sequential(
            
            # first block
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                        
            # second block
            torch.nn.Conv2d(64, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 192, kernel_size=3),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.inception_3a_1 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_3a_2 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 96, kernel_size=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(96, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_3a_3 = torch.nn.Sequential(
            torch.nn.Conv2d(192, 16, kernel_size=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_3a_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(192, 32, kernel_size=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_3b_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_3b_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_3b_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 32, kernel_size=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 96, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_3b_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(256, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4a_1 = torch.nn.Sequential(
            torch.nn.Conv2d(480, 192, kernel_size=1),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4a_2 = torch.nn.Sequential(
            torch.nn.Conv2d(480, 96, kernel_size=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(96, 208, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(208),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4a_3 = torch.nn.Sequential(
            torch.nn.Conv2d(480, 16, kernel_size=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 48, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4a_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(480, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4b_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 160, kernel_size=1),
            torch.nn.BatchNorm2d(160),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4b_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 112, kernel_size=1),
            torch.nn.BatchNorm2d(112),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(112, 224, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(224),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4b_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 24, kernel_size=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4b_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(512, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4c_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4c_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4c_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 24, kernel_size=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(24, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4c_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(512, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4d_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 112, kernel_size=1),
            torch.nn.BatchNorm2d(112),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4d_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 144, kernel_size=1),
            torch.nn.BatchNorm2d(144),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(144, 288, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(288),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4d_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 32, kernel_size=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4d_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(512, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4e_1 = torch.nn.Sequential(
            torch.nn.Conv2d(528, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4e_2 = torch.nn.Sequential(
            torch.nn.Conv2d(528, 160, kernel_size=1),
            torch.nn.BatchNorm2d(160),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(160, 320, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(320),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4e_3 = torch.nn.Sequential(
            torch.nn.Conv2d(528, 32, kernel_size=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 128, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_4e_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(528, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5a_1 = torch.nn.Sequential(
            torch.nn.Conv2d(832, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5a_2 = torch.nn.Sequential(
            torch.nn.Conv2d(832, 160, kernel_size=1),
            torch.nn.BatchNorm2d(160),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(160, 320, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(320),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5a_3 = torch.nn.Sequential(
            torch.nn.Conv2d(832, 32, kernel_size=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 128, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5a_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(832, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5b_1 = torch.nn.Sequential(
            torch.nn.Conv2d(832, 384, kernel_size=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5b_2 = torch.nn.Sequential(
            torch.nn.Conv2d(832, 192, kernel_size=1),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5b_3 = torch.nn.Sequential(
            torch.nn.Conv2d(832, 48, kernel_size=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(inplace=True),            
            torch.nn.Conv2d(48, 128, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.inception_5b_4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(832, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d(7)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(1024 * 7 * 7, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
        
        self.max_pooling = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x1 = self.inception_3a_1(x)
        x2 = self.inception_3a_2(x)
        x3 = self.inception_3a_3(x)
        x4 = self.inception_3a_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x1 = self.inception_3b_1(x)
        x2 = self.inception_3b_2(x)
        x3 = self.inception_3b_3(x)
        x4 = self.inception_3b_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.max_pooling(x)
        x1 = self.inception_4a_1(x)
        x2 = self.inception_4a_2(x)
        x3 = self.inception_4a_3(x)
        x4 = self.inception_4a_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x1 = self.inception_4b_1(x)
        x2 = self.inception_4b_2(x)
        x3 = self.inception_4b_3(x)
        x4 = self.inception_4b_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x1 = self.inception_4c_1(x)
        x2 = self.inception_4c_2(x)
        x3 = self.inception_4c_3(x)
        x4 = self.inception_4c_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x1 = self.inception_4d_1(x)
        x2 = self.inception_4d_2(x)
        x3 = self.inception_4d_3(x)
        x4 = self.inception_4d_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x1 = self.inception_4e_1(x)
        x2 = self.inception_4e_2(x)
        x3 = self.inception_4e_3(x)
        x4 = self.inception_4e_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.max_pooling(x)
        x1 = self.inception_5a_1(x)
        x2 = self.inception_5a_2(x)
        x3 = self.inception_5a_3(x)
        x4 = self.inception_5a_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x1 = self.inception_5b_1(x)
        x2 = self.inception_5b_2(x)
        x3 = self.inception_5b_3(x)
        x4 = self.inception_5b_4(x)
        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
