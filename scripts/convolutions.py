# header files 
import numpy as np
import torch
import torch.nn


# Convolution class
class Convolution(torch.nn.Sequential):
    
    # init method
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.add_module("conv", torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.strides, padding=self.padding))
        self.add_module("norm", torch.nn.BatchNorm2d(self.out_channels))
        self.add_module("act", torch.nn.ReLU(inplace=True))    
