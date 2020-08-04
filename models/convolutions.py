# header files 
import numpy as np
import torch
import torch.nn as nn


class Convolution(nn.Sequential):
    
    # init method
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        strides: int = 1,
        padding: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.add_module("conv", Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.strides, padding=self.padding))
        self.add_module("norm", BatchNorm2d(self.out_channels))
        self.add_module("act", ReLU(inplace=True))    
