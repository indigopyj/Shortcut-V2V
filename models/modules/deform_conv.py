import torch
import torchvision.ops
from torch import nn
import math

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 num_deformable_groups=1,
                 stride=1,
                 padding=1,
                 bias=False,):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        self.kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.in_channels = in_channels
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * self.kernel_size[0] * self.kernel_size[1] * num_deformable_groups,
                                     kernel_size=self.kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=self.kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x, offset=None):
        if offset is None:
            offset = self.offset_conv(x)
        if offset.shape[1] != 2 * self.kernel_size[0] * self.kernel_size[1]:
            offset = offset.repeat(1, 3 * 3, 1, 1)
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          stride=self.stride,
                                          mask=None
                                          )
        return x