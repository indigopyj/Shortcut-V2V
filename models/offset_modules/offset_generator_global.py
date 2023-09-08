import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from models.modules.deform_conv import DeformableConv2d
from models.modules.HetConv_module import HetConv
import torch.nn.functional as F


class GlobalOffsetGenerator(nn.Module):
    def __init__(self, opt, input_nc, ngf, output_nc, n_layers=1):
        super(GlobalOffsetGenerator, self).__init__()
        self.opt=opt
        self.model = []
        for l in range(n_layers):
            if l != 0: input_nc = ngf
            self.model += [HetConv(input_nc, ngf, padding=1),
                                    nn.ReLU(inplace=True)]
        
        self.model = nn.Sequential(*self.model)
        self.offset_conv = nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
            
        
    def forward(self, img1, img2):
        output = self.model(torch.cat((img2, img1), dim=1))
        global_offset = self.offset_conv(output)
        return global_offset