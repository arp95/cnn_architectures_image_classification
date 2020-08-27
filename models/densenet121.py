# header files
import torch
import torch.nn as nn
import torchvision
import numpy as np


# define network (remember input size: (224 x 224 x 3))
class DenseNet_121(torch.nn.Module):
    
    # define dense block
    def dense_block(self, input_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 128, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
    
    # init function
    def __init__(self, num_classes = 2):
        super(DenseNet_121, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # dense block 1 (56 x 56 x 64)
        self.dense_block_1_1 = self.dense_block(64)
        self.dense_block_1_2 = self.dense_block(96)
        self.dense_block_1_3 = self.dense_block(128)
        self.dense_block_1_4 = self.dense_block(160)
        self.dense_block_1_5 = self.dense_block(192)
        self.dense_block_1_6 = self.dense_block(224)
        
        # transition block 1
        self.transition_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1, bias=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # dense block 2 (28 x 28 x 128)
        self.dense_block_2_1 = self.dense_block(128)
        self.dense_block_2_2 = self.dense_block(160)
        self.dense_block_2_3 = self.dense_block(192)
        self.dense_block_2_4 = self.dense_block(224)
        self.dense_block_2_5 = self.dense_block(256)
        self.dense_block_2_6 = self.dense_block(288)
        self.dense_block_2_7 = self.dense_block(320)
        self.dense_block_2_8 = self.dense_block(352)
        self.dense_block_2_9 = self.dense_block(384)
        self.dense_block_2_10 = self.dense_block(416)
        self.dense_block_2_11 = self.dense_block(448)
        self.dense_block_2_12 = self.dense_block(480)
        
        
        # transition block 2
        self.transition_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1, bias=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # dense block 3 (14 x 14 x 240)
        self.dense_block_3_1 = self.dense_block(256)
        self.dense_block_3_2 = self.dense_block(288)
        self.dense_block_3_3 = self.dense_block(320)
        self.dense_block_3_4 = self.dense_block(352)
        self.dense_block_3_5 = self.dense_block(384)
        self.dense_block_3_6 = self.dense_block(416)
        self.dense_block_3_7 = self.dense_block(448)
        self.dense_block_3_8 = self.dense_block(480)
        self.dense_block_3_9 = self.dense_block(512)
        self.dense_block_3_10 = self.dense_block(544)
        self.dense_block_3_11 = self.dense_block(576)
        self.dense_block_3_12 = self.dense_block(608)
        self.dense_block_3_13 = self.dense_block(640)
        self.dense_block_3_14 = self.dense_block(672)
        self.dense_block_3_15 = self.dense_block(704)
        self.dense_block_3_16 = self.dense_block(736)
        self.dense_block_3_17 = self.dense_block(768)
        self.dense_block_3_18 = self.dense_block(800)
        self.dense_block_3_19 = self.dense_block(832)
        self.dense_block_3_20 = self.dense_block(864)
        self.dense_block_3_21 = self.dense_block(896)
        self.dense_block_3_22 = self.dense_block(928)
        self.dense_block_3_23 = self.dense_block(960)
        self.dense_block_3_24 = self.dense_block(992)
        
        
        # transition block 3
        self.transition_block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # dense block 4 (7 x 7 x 512)
        self.dense_block_4_1 = self.dense_block(512)
        self.dense_block_4_2 = self.dense_block(544)
        self.dense_block_4_3 = self.dense_block(576)
        self.dense_block_4_4 = self.dense_block(608)
        self.dense_block_4_5 = self.dense_block(640)
        self.dense_block_4_6 = self.dense_block(672)
        self.dense_block_4_7 = self.dense_block(704)
        self.dense_block_4_8 = self.dense_block(736)
        self.dense_block_4_9 = self.dense_block(768)
        self.dense_block_4_10 = self.dense_block(800)
        self.dense_block_4_11 = self.dense_block(832)
        self.dense_block_4_12 = self.dense_block(864)
        self.dense_block_4_13 = self.dense_block(896)
        self.dense_block_4_14 = self.dense_block(928)
        self.dense_block_4_15 = self.dense_block(960)
        self.dense_block_4_16 = self.dense_block(992)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d(7)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024 * 7 * 7, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        
        # dense block 1
        x_1 = self.dense_block_1_1(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_1_2(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_1_3(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_1_4(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_1_5(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_1_6(x)
        x = torch.cat([x, x_1], 1)
        
        # transition block 1
        x = self.transition_block_1(x)
        
        # dense block 2
        x_1 = self.dense_block_2_1(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_2(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_3(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_4(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_5(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_6(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_7(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_8(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_9(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_10(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_11(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_2_12(x)
        x = torch.cat([x, x_1], 1)
        
        # transition block 2
        x = self.transition_block_2(x)
        
        # dense block 3
        x_1 = self.dense_block_3_1(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_2(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_3(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_4(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_5(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_6(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_7(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_8(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_9(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_10(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_11(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_12(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_13(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_14(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_15(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_16(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_17(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_18(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_19(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_20(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_21(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_22(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_23(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_3_24(x)
        x = torch.cat([x, x_1], 1)
        
        # transition block 3
        x = self.transition_block_3(x)
        
        # dense block 4
        x_1 = self.dense_block_4_1(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_2(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_3(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_4(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_5(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_6(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_7(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_8(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_9(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_10(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_11(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_12(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_13(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_14(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_15(x)
        x = torch.cat([x, x_1], 1)
        x_1 = self.dense_block_4_16(x)
        x = torch.cat([x, x_1], 1)
        
        x = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
